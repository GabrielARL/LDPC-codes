/* JOINT_DFEC_BP.C - Joint short-ISI channel estimation and BP decoding.
 *
 * This is a real-valued, BPSK-oriented counterpart to the DFEC experiment's
 * alternating channel/symbol optimizer. It assumes a short contiguous ISI
 * channel with taps provided on the command line as a prior estimate. For
 * each received block, it:
 *
 *   1. Builds a DFE-style warm start from the tap prior.
 *   2. Alternates between ridge-regularized tap estimation and gradient-based
 *      optimization of soft symbol logits z, with x = tanh(z).
 *   3. Converts z to likelihood ratios and runs the existing PRPRP decoder.
 *
 * The parity penalty operates on BPSK symbols using the actual target product
 * implied by each parity-check row's degree.
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "alloc.h"
#include "blockio.h"
#include "mod2sparse.h"
#include "mod2dense.h"
#include "mod2convert.h"
#include "check.h"
#include "dec.h"
#include "open.h"
#include "rcode.h"

#define DEFAULT_BP_ITERS   200
#define DEFAULT_ALT_ITERS  4
#define DEFAULT_Z_STEPS    50
#define DEFAULT_RESTARTS   2
#define DEFAULT_LAMBDA     2.0
#define DEFAULT_GAMMA      1e-3
#define DEFAULT_ETA        1.0
#define DEFAULT_WARM_CLIP  0.98
#define DEFAULT_Z_LIMIT    10.0

#define ARMIJO_C1          1e-4
#define MIN_STEP_SIZE      1e-6
#define MAX_ABS_LLR        60.0
#define SOLVER_EPS         1e-10

typedef struct
{
  int n_checks;
  int max_row_weight;
  int *offsets;
  int *indices;
  double *targets;
} parity_info;

typedef struct
{
  int bp_iters;
  int alt_iters;
  int z_steps;
  int restarts;
  int summary_table;
  double lambda;
  double gamma;
  double eta;
  double warm_clip;
  double z_limit;
  char *prob_file;
  char *grad_file;
  char *eval_z_file;
  char *eval_h_file;
} options;

typedef struct
{
  int n;
  int n_taps;
  double *z0;
  double *z;
  double *z_trial;
  double *best_z;
  double *x;
  double *res;
  double *grad;
  double *h;
  double *best_h;
  double *gram;
  double *rhs;
  double *prefix;
  double *suffix;
  char *direct_bits;
  char *direct_pchk;
} joint_workspace;

typedef struct
{
  double objective;
  double data_loss;
  double parity_loss;
  unsigned bp_iters;
  int restart_idx;
  int direct_valid;
  int bp_valid;
  double joint_sec;
  double bp_sec;
  double total_sec;
} block_stats;

static void usage(void);
static int parse_int_arg(const char *text, int *value);
static int parse_double_arg(const char *text, double *value);
static double clamp_double(double value, double lo, double hi);
static double safe_exp(double value);
static double now_seconds(void);
static void init_options(options *opts);
static void build_parity_info(mod2sparse *matrix, parity_info *info);
static int read_received_block(FILE *rf, double *y, int n);
static void read_double_vector_file(const char *path, double *values, int count,
                                    const char *label);
static void write_double_vector(FILE *f, const double *values, int count);
static void write_probabilities(FILE *pf, const double *bitpr, int n);
static void dfe_warm_start(const double *y, int n, const double *taps, int n_taps,
                           double clip, double *equalized, double *decisions, double *z0);
static int solve_linear_system(double *matrix, double *rhs, int n);
static void solve_supported_channel(const double *x, const double *y, int n,
                                    const double *h_prior, int n_taps,
                                    double gamma, double eta,
                                    double *gram, double *rhs, double *h_out);
static double parity_loss_and_grad(const parity_info *info, const double *x,
                                   double lambda, double *grad_z,
                                   double *prefix, double *suffix);
static double evaluate_z_objective(const parity_info *info, const double *y,
                                   const double *h, const double *z,
                                   int n, int n_taps,
                                   double lambda, double gamma,
                                   double *x, double *res, double *grad,
                                   double *prefix, double *suffix,
                                   double *data_loss_out,
                                   double *parity_loss_out);
static double evaluate_full_objective(const parity_info *info, const double *y,
                                      const double *h, const double *h_prior,
                                      const double *z, int n, int n_taps,
                                      double lambda, double gamma, double eta,
                                      double *x, double *res,
                                      double *prefix, double *suffix,
                                      double *data_loss_out,
                                      double *parity_loss_out);
static void optimize_z(const parity_info *info, const double *y, const double *h,
                       int n, int n_taps, const options *opts,
                       double *z, joint_workspace *ws);
static void hard_decision_from_z(const double *z, int n, char *bits);
static double restart_scale(int restart_idx);
static void run_joint_decoder(mod2sparse *matrix, const parity_info *info,
                              const options *opts, const double *y,
                              const double *h_prior, joint_workspace *ws,
                              double *lratio, char *dblk, char *pchk,
                              double *bitpr, block_stats *stats);

int main(int argc, char **argv)
{
  char *pchk_file;
  char *received_file;
  char *decoded_file;
  FILE *rf;
  FILE *df;
  FILE *pf;
  FILE *gf;
  parity_info info;
  joint_workspace ws;
  options opts;
  double *y;
  double *h_prior;
  double *eval_z;
  double *eval_h;
  double *lratio;
  double *bitpr;
  char *dblk;
  char *pchk;
  int argi;
  int positional_count;
  char *positional[3];
  int block_count;
  int valid_count;
  double total_objective;
  double total_bp_iters;

  init_options(&opts);

  positional_count = 0;
  argi = 1;

  while (argi < argc && positional_count < 3)
  {
    if (strcmp(argv[argi], "-f") == 0)
    {
      blockio_flush = 1;
      argi += 1;
    }
    else if (strcmp(argv[argi], "-t") == 0)
    {
      opts.summary_table = 1;
      argi += 1;
    }
    else if (strcmp(argv[argi], "-T") == 0)
    {
      table = 2;
      argi += 1;
    }
    else if (strcmp(argv[argi], "-p") == 0)
    {
      if (argi + 1 >= argc)
      {
        usage();
      }
      opts.prob_file = argv[argi + 1];
      argi += 2;
    }
    else if (strcmp(argv[argi], "-G") == 0)
    {
      if (argi + 1 >= argc)
      {
        usage();
      }
      opts.grad_file = argv[argi + 1];
      argi += 2;
    }
    else if (strcmp(argv[argi], "-Z") == 0)
    {
      if (argi + 1 >= argc)
      {
        usage();
      }
      opts.eval_z_file = argv[argi + 1];
      argi += 2;
    }
    else if (strcmp(argv[argi], "-H") == 0)
    {
      if (argi + 1 >= argc)
      {
        usage();
      }
      opts.eval_h_file = argv[argi + 1];
      argi += 2;
    }
    else if (strcmp(argv[argi], "-b") == 0)
    {
      if (argi + 1 >= argc || !parse_int_arg(argv[argi + 1], &opts.bp_iters)
                          || opts.bp_iters <= 0)
      {
        usage();
      }
      argi += 2;
    }
    else if (strcmp(argv[argi], "-o") == 0)
    {
      if (argi + 1 >= argc || !parse_int_arg(argv[argi + 1], &opts.alt_iters)
                          || opts.alt_iters <= 0)
      {
        usage();
      }
      argi += 2;
    }
    else if (strcmp(argv[argi], "-j") == 0)
    {
      if (argi + 1 >= argc || !parse_int_arg(argv[argi + 1], &opts.z_steps)
                          || opts.z_steps <= 0)
      {
        usage();
      }
      argi += 2;
    }
    else if (strcmp(argv[argi], "-r") == 0)
    {
      if (argi + 1 >= argc || !parse_int_arg(argv[argi + 1], &opts.restarts)
                          || opts.restarts <= 0)
      {
        usage();
      }
      argi += 2;
    }
    else if (strcmp(argv[argi], "-l") == 0)
    {
      if (argi + 1 >= argc || !parse_double_arg(argv[argi + 1], &opts.lambda)
                          || opts.lambda < 0.0)
      {
        usage();
      }
      argi += 2;
    }
    else if (strcmp(argv[argi], "-g") == 0)
    {
      if (argi + 1 >= argc || !parse_double_arg(argv[argi + 1], &opts.gamma)
                          || opts.gamma < 0.0)
      {
        usage();
      }
      argi += 2;
    }
    else if (strcmp(argv[argi], "-e") == 0)
    {
      if (argi + 1 >= argc || !parse_double_arg(argv[argi + 1], &opts.eta)
                          || opts.eta < 0.0)
      {
        usage();
      }
      argi += 2;
    }
    else if (strcmp(argv[argi], "-c") == 0)
    {
      if (argi + 1 >= argc || !parse_double_arg(argv[argi + 1], &opts.warm_clip)
                          || opts.warm_clip <= 0.0 || opts.warm_clip >= 1.0)
      {
        usage();
      }
      argi += 2;
    }
    else
    {
      positional[positional_count++] = argv[argi];
      argi += 1;
    }
  }

  if (positional_count != 3 || argi >= argc)
  {
    usage();
  }

  pchk_file = positional[0];
  received_file = positional[1];
  decoded_file = positional[2];

  if ((strcmp(pchk_file, "-") == 0) + (strcmp(received_file, "-") == 0) > 1)
  {
    fprintf(stderr, "Can't read more than one stream from standard input\n");
    exit(1);
  }

  if ((table > 0)
   + (strcmp(decoded_file, "-") == 0)
   + (opts.prob_file != 0 && strcmp(opts.prob_file, "-") == 0)
   + (opts.grad_file != 0 && strcmp(opts.grad_file, "-") == 0) > 1)
  {
    fprintf(stderr, "Can't send more than one stream to standard output\n");
    exit(1);
  }

  if (opts.grad_file == 0
   && (opts.eval_z_file != 0 || opts.eval_h_file != 0))
  {
    fprintf(stderr, "Options -Z and -H require -G\n");
    exit(1);
  }

  if (opts.gamma + opts.eta <= 0.0)
  {
    fprintf(stderr, "At least one of gamma or eta must be positive\n");
    exit(1);
  }

  read_pchk(pchk_file);

  if (N <= M)
  {
    fprintf(stderr,
      "Number of bits (%d) should be greater than number of checks (%d)\n", N, M);
    exit(1);
  }

  ws.n_taps = argc - argi;
  if (ws.n_taps <= 0)
  {
    usage();
  }

  h_prior = chk_alloc(ws.n_taps, sizeof *h_prior);
  for (int i = 0; i < ws.n_taps; i++)
  {
    if (!parse_double_arg(argv[argi + i], &h_prior[i]))
    {
      usage();
    }
  }

  if (fabs(h_prior[0]) < SOLVER_EPS)
  {
    fprintf(stderr, "First tap must be non-zero for the DFE warm start\n");
    exit(1);
  }

  rf = open_file_std(received_file, "r");
  if (rf == 0)
  {
    fprintf(stderr, "Can't open file of received data: %s\n", received_file);
    exit(1);
  }

  df = open_file_std(decoded_file, "w");
  if (df == 0)
  {
    fprintf(stderr, "Can't create file for decoded data: %s\n", decoded_file);
    exit(1);
  }

  pf = 0;
  if (opts.prob_file != 0)
  {
    pf = open_file_std(opts.prob_file, "w");
    if (pf == 0)
    {
      fprintf(stderr, "Can't create file for bit probabilities: %s\n", opts.prob_file);
      exit(1);
    }
  }

  gf = 0;
  if (opts.grad_file != 0)
  {
    gf = open_file_std(opts.grad_file, "w");
    if (gf == 0)
    {
      fprintf(stderr, "Can't create file for objective gradient: %s\n", opts.grad_file);
      exit(1);
    }
  }

  build_parity_info(H, &info);

  ws.n = N;
  ws.z0 = chk_alloc(N, sizeof *ws.z0);
  ws.z = chk_alloc(N, sizeof *ws.z);
  ws.z_trial = chk_alloc(N, sizeof *ws.z_trial);
  ws.best_z = chk_alloc(N, sizeof *ws.best_z);
  ws.x = chk_alloc(N, sizeof *ws.x);
  ws.res = chk_alloc(N, sizeof *ws.res);
  ws.grad = chk_alloc(N, sizeof *ws.grad);
  ws.h = chk_alloc(ws.n_taps, sizeof *ws.h);
  ws.best_h = chk_alloc(ws.n_taps, sizeof *ws.best_h);
  ws.gram = chk_alloc(ws.n_taps * ws.n_taps, sizeof *ws.gram);
  ws.rhs = chk_alloc(ws.n_taps, sizeof *ws.rhs);
  ws.prefix = chk_alloc(info.max_row_weight > 0 ? info.max_row_weight : 1,
                        sizeof *ws.prefix);
  ws.suffix = chk_alloc(info.max_row_weight > 0 ? info.max_row_weight : 1,
                        sizeof *ws.suffix);
  ws.direct_bits = chk_alloc(N, sizeof *ws.direct_bits);
  ws.direct_pchk = chk_alloc(M, sizeof *ws.direct_pchk);

  y = chk_alloc(N, sizeof *y);
  eval_z = opts.eval_z_file != 0 ? chk_alloc(N, sizeof *eval_z) : 0;
  eval_h = opts.eval_h_file != 0 ? chk_alloc(ws.n_taps, sizeof *eval_h) : 0;
  lratio = chk_alloc(N, sizeof *lratio);
  bitpr = chk_alloc(N, sizeof *bitpr);
  dblk = chk_alloc(N, sizeof *dblk);
  pchk = chk_alloc(M, sizeof *pchk);

  if (opts.eval_z_file != 0)
  {
    read_double_vector_file(opts.eval_z_file, eval_z, N, "evaluation z");
  }
  if (opts.eval_h_file != 0)
  {
    read_double_vector_file(opts.eval_h_file, eval_h, ws.n_taps, "evaluation taps");
  }

  max_iter = opts.bp_iters;
  prprp_decode_setup();

  if (opts.summary_table)
  {
    fprintf(stderr,
      "block restart direct-valid bp-valid bp-iters objective data-loss parity-loss "
      "joint-sec bp-sec total-sec taps\n");
  }

  block_count = 0;
  valid_count = 0;
  total_objective = 0.0;
  total_bp_iters = 0.0;

  while (read_received_block(rf, y, N))
  {
    block_stats stats;

    block_no = block_count;
    run_joint_decoder(H, &info, &opts, y, h_prior, &ws, lratio, dblk, pchk, bitpr, &stats);

    blockio_write(df, dblk, N);
    if (pf != 0)
    {
      write_probabilities(pf, bitpr, N);
    }
    if (gf != 0)
    {
      const double *grad_z = eval_z != 0 ? eval_z : ws.best_z;
      const double *grad_h = eval_h != 0 ? eval_h : ws.best_h;

      evaluate_z_objective(&info, y, grad_h, grad_z, ws.n, ws.n_taps,
                           opts.lambda, opts.gamma,
                           ws.x, ws.res, ws.grad,
                           ws.prefix, ws.suffix, 0, 0);
      write_double_vector(gf, ws.grad, ws.n);
    }

    if (opts.summary_table)
    {
      int i;

      fprintf(stderr, "%5d %7d %12d %8d %8u %.6e %.6e %.6e %.6e %.6e %.6e [",
              block_count, stats.restart_idx, stats.direct_valid, stats.bp_valid,
              stats.bp_iters, stats.objective, stats.data_loss, stats.parity_loss,
              stats.joint_sec, stats.bp_sec, stats.total_sec);
      for (i = 0; i < ws.n_taps; i++)
      {
        fprintf(stderr, "%s%.6f", i == 0 ? "" : ", ", ws.best_h[i]);
      }
      fprintf(stderr, "]\n");
    }

    valid_count += stats.bp_valid;
    total_objective += stats.objective;
    total_bp_iters += stats.bp_iters;
    block_count += 1;

    if (ferror(df) || (pf != 0 && ferror(pf)))
    {
      break;
    }
  }

  fflush(stdout);

  if (block_count == 0)
  {
    fprintf(stderr, "No complete received blocks were found\n");
  }
  else
  {
    fprintf(stderr,
      "Joint DFEC-BP decoded %d blocks, %d valid. Average %.1f BP iterations, objective %.3e\n",
      block_count, valid_count, total_bp_iters / block_count,
      total_objective / block_count);
  }

  if (ferror(df) || fclose(df) != 0)
  {
    fprintf(stderr, "Error writing decoded blocks to %s\n", decoded_file);
    exit(1);
  }

  if (pf != 0)
  {
    if (ferror(pf) || fclose(pf) != 0)
    {
      fprintf(stderr, "Error writing bit probabilities to %s\n", opts.prob_file);
      exit(1);
    }
  }

  if (gf != 0)
  {
    if (ferror(gf) || fclose(gf) != 0)
    {
      fprintf(stderr, "Error writing objective gradients to %s\n", opts.grad_file);
      exit(1);
    }
  }

  if (fclose(rf) != 0)
  {
    fprintf(stderr, "Error closing received data file: %s\n", received_file);
    exit(1);
  }

  return 0;
}

static void usage(void)
{
  fprintf(stderr, "Usage:\n");
  fprintf(stderr,
    "  joint_dfec_bp [options] pchk-file received-file decoded-file tap1 [tap2 ...]\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -p file   Write posterior P(bit=1) values to file\n");
  fprintf(stderr, "  -G file   Write d objective / dz values to file\n");
  fprintf(stderr, "  -Z file   Read z values for gradient evaluation from file\n");
  fprintf(stderr, "  -H file   Read tap values for gradient evaluation from file\n");
  fprintf(stderr, "  -f        Flush decoded blocks immediately\n");
  fprintf(stderr, "  -t        Print one-line block summaries to stderr\n");
  fprintf(stderr, "  -T        Enable detailed BP trace output\n");
  fprintf(stderr, "  -b iters  BP iterations (default %d)\n", DEFAULT_BP_ITERS);
  fprintf(stderr, "  -o iters  Alternating outer iterations (default %d)\n", DEFAULT_ALT_ITERS);
  fprintf(stderr, "  -j iters  Gradient steps per outer iteration (default %d)\n", DEFAULT_Z_STEPS);
  fprintf(stderr, "  -r count  Restarts (default %d)\n", DEFAULT_RESTARTS);
  fprintf(stderr, "  -l value  Parity penalty weight lambda (default %.3g)\n", DEFAULT_LAMBDA);
  fprintf(stderr, "  -g value  Regularization gamma (default %.3g)\n", DEFAULT_GAMMA);
  fprintf(stderr, "  -e value  Channel-prior weight eta (default %.3g)\n", DEFAULT_ETA);
  fprintf(stderr, "  -c value  Warm-start clip in (0,1) (default %.2f)\n", DEFAULT_WARM_CLIP);
  exit(1);
}

static int parse_int_arg(const char *text, int *value)
{
  char extra;
  return sscanf(text, "%d%c", value, &extra) == 1;
}

static int parse_double_arg(const char *text, double *value)
{
  char extra;
  return sscanf(text, "%lf%c", value, &extra) == 1;
}

static double clamp_double(double value, double lo, double hi)
{
  if (value < lo)
  {
    return lo;
  }
  if (value > hi)
  {
    return hi;
  }
  return value;
}

static double safe_exp(double value)
{
  return exp(clamp_double(value, -MAX_ABS_LLR, MAX_ABS_LLR));
}

static double now_seconds(void)
{
  struct timespec ts;

  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
  {
    fprintf(stderr, "clock_gettime failed\n");
    exit(1);
  }

  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void init_options(options *opts)
{
  blockio_flush = 0;
  table = 0;

  opts->bp_iters = DEFAULT_BP_ITERS;
  opts->alt_iters = DEFAULT_ALT_ITERS;
  opts->z_steps = DEFAULT_Z_STEPS;
  opts->restarts = DEFAULT_RESTARTS;
  opts->summary_table = 0;
  opts->lambda = DEFAULT_LAMBDA;
  opts->gamma = DEFAULT_GAMMA;
  opts->eta = DEFAULT_ETA;
  opts->warm_clip = DEFAULT_WARM_CLIP;
  opts->z_limit = DEFAULT_Z_LIMIT;
  opts->prob_file = 0;
  opts->grad_file = 0;
  opts->eval_z_file = 0;
  opts->eval_h_file = 0;
}

static void build_parity_info(mod2sparse *matrix, parity_info *info)
{
  int edge_count;
  int cursor;

  info->n_checks = mod2sparse_rows(matrix);
  info->max_row_weight = 0;
  info->offsets = chk_alloc(info->n_checks + 1, sizeof *info->offsets);
  info->targets = chk_alloc(info->n_checks, sizeof *info->targets);

  edge_count = 0;
  info->offsets[0] = 0;

  for (int row = 0; row < info->n_checks; row++)
  {
    int degree = mod2sparse_count_row(matrix, row);

    edge_count += degree;
    info->offsets[row + 1] = edge_count;
    info->targets[row] = degree % 2 == 0 ? 1.0 : -1.0;
    if (degree > info->max_row_weight)
    {
      info->max_row_weight = degree;
    }
  }

  info->indices = chk_alloc(edge_count > 0 ? edge_count : 1, sizeof *info->indices);

  cursor = 0;
  for (int row = 0; row < info->n_checks; row++)
  {
    mod2entry *entry;

    for (entry = mod2sparse_first_in_row(matrix, row);
         !mod2sparse_at_end(entry);
         entry = mod2sparse_next_in_row(entry))
    {
      info->indices[cursor++] = mod2sparse_col(entry);
    }
  }
}

static int read_received_block(FILE *rf, double *y, int n)
{
  int i;

  for (i = 0; i < n; i++)
  {
    int rc = fscanf(rf, "%lf", &y[i]);

    if (rc == EOF)
    {
      if (i > 0)
      {
        fprintf(stderr,
          "Warning: Short block (%d long) at end of received file ignored\n", i);
      }
      return 0;
    }
    if (rc != 1)
    {
      fprintf(stderr, "File of received data is garbled\n");
      exit(1);
    }
  }

  return 1;
}

static void read_double_vector_file(const char *path, double *values, int count,
                                    const char *label)
{
  FILE *f;
  int i;

  f = open_file_std((char *) path, "r");
  if (f == 0)
  {
    fprintf(stderr, "Can't open %s file: %s\n", label, path);
    exit(1);
  }

  for (i = 0; i < count; i++)
  {
    if (fscanf(f, "%lf", &values[i]) != 1)
    {
      fprintf(stderr, "%s file %s does not contain %d doubles\n", label, path, count);
      exit(1);
    }
  }

  if (fclose(f) != 0)
  {
    fprintf(stderr, "Error closing %s file: %s\n", label, path);
    exit(1);
  }
}

static void write_double_vector(FILE *f, const double *values, int count)
{
  for (int i = 0; i < count; i++)
  {
    fprintf(f, "%s%.12e", i == 0 ? "" : " ", values[i]);
  }
  fprintf(f, "\n");
}

static void write_probabilities(FILE *pf, const double *bitpr, int n)
{
  for (int i = 0; i < n; i++)
  {
    fprintf(pf, " %.5f", bitpr[i]);
  }
  fprintf(pf, "\n");
}

static void dfe_warm_start(const double *y, int n, const double *taps, int n_taps,
                           double clip, double *equalized, double *decisions, double *z0)
{
  for (int i = 0; i < n; i++)
  {
    double fb = 0.0;
    double eq;
    double soft;

    for (int j = 1; j < n_taps; j++)
    {
      int idx = i - j;
      if (idx >= 0)
      {
        fb += taps[j] * decisions[idx];
      }
    }

    eq = (y[i] - fb) / taps[0];
    equalized[i] = eq;
    decisions[i] = eq >= 0.0 ? 1.0 : -1.0;

    soft = clamp_double(eq, -clip, clip);
    z0[i] = atanh(soft);
  }
}

static int solve_linear_system(double *matrix, double *rhs, int n)
{
  for (int col = 0; col < n; col++)
  {
    int pivot = col;
    double pivot_abs = fabs(matrix[col * n + col]);

    for (int row = col + 1; row < n; row++)
    {
      double cand = fabs(matrix[row * n + col]);
      if (cand > pivot_abs)
      {
        pivot = row;
        pivot_abs = cand;
      }
    }

    if (pivot_abs < SOLVER_EPS)
    {
      return 0;
    }

    if (pivot != col)
    {
      double tmp;

      for (int j = col; j < n; j++)
      {
        tmp = matrix[col * n + j];
        matrix[col * n + j] = matrix[pivot * n + j];
        matrix[pivot * n + j] = tmp;
      }

      tmp = rhs[col];
      rhs[col] = rhs[pivot];
      rhs[pivot] = tmp;
    }

    for (int row = col + 1; row < n; row++)
    {
      double factor = matrix[row * n + col] / matrix[col * n + col];

      matrix[row * n + col] = 0.0;
      for (int j = col + 1; j < n; j++)
      {
        matrix[row * n + j] -= factor * matrix[col * n + j];
      }
      rhs[row] -= factor * rhs[col];
    }
  }

  for (int row = n - 1; row >= 0; row--)
  {
    double sum = rhs[row];

    for (int j = row + 1; j < n; j++)
    {
      sum -= matrix[row * n + j] * rhs[j];
    }

    rhs[row] = sum / matrix[row * n + row];
  }

  return 1;
}

static void solve_supported_channel(const double *x, const double *y, int n,
                                    const double *h_prior, int n_taps,
                                    double gamma, double eta,
                                    double *gram, double *rhs, double *h_out)
{
  int ok;

  for (int row = 0; row < n_taps; row++)
  {
    rhs[row] = eta * h_prior[row];

    for (int col = 0; col < n_taps; col++)
    {
      double sum = row == col ? gamma + eta : 0.0;
      int start = row > col ? row : col;

      for (int i = start; i < n; i++)
      {
        sum += x[i - row] * x[i - col];
      }

      gram[row * n_taps + col] = sum;
    }

    for (int i = row; i < n; i++)
    {
      rhs[row] += x[i - row] * y[i];
    }
  }

  ok = solve_linear_system(gram, rhs, n_taps);
  if (!ok)
  {
    for (int i = 0; i < n_taps; i++)
    {
      gram[i * n_taps + i] += 1e-6;
    }
    ok = solve_linear_system(gram, rhs, n_taps);
  }

  if (!ok)
  {
    fprintf(stderr, "Failed to solve for supported channel taps\n");
    exit(1);
  }

  for (int i = 0; i < n_taps; i++)
  {
    h_out[i] = rhs[i];
  }
}

static double parity_loss_and_grad(const parity_info *info, const double *x,
                                   double lambda, double *grad_z,
                                   double *prefix, double *suffix)
{
  double loss = 0.0;

  for (int row = 0; row < info->n_checks; row++)
  {
    int start = info->offsets[row];
    int end = info->offsets[row + 1];
    int len = end - start;
    double prod = 1.0;
    double diff;

    for (int k = 0; k < len; k++)
    {
      prod *= x[info->indices[start + k]];
    }

    diff = prod - info->targets[row];
    loss += diff * diff;

    if (grad_z == 0 || len == 0)
    {
      continue;
    }

    prefix[0] = 1.0;
    for (int k = 1; k < len; k++)
    {
      prefix[k] = prefix[k - 1] * x[info->indices[start + k - 1]];
    }

    suffix[len - 1] = 1.0;
    for (int k = len - 2; k >= 0; k--)
    {
      suffix[k] = suffix[k + 1] * x[info->indices[start + k + 1]];
    }

    for (int k = 0; k < len; k++)
    {
      int col = info->indices[start + k];
      double prod_others = prefix[k] * suffix[k];
      double sech2 = 1.0 - x[col] * x[col];

      grad_z[col] += lambda * 2.0 * diff * prod_others * sech2;
    }
  }

  return loss;
}

static double evaluate_z_objective(const parity_info *info, const double *y,
                                   const double *h, const double *z,
                                   int n, int n_taps,
                                   double lambda, double gamma,
                                   double *x, double *res, double *grad,
                                   double *prefix, double *suffix,
                                   double *data_loss_out,
                                   double *parity_loss_out)
{
  double data_loss;
  double parity_loss;
  double reg_loss;

  for (int i = 0; i < n; i++)
  {
    x[i] = tanh(z[i]);
  }

  data_loss = 0.0;
  for (int i = 0; i < n; i++)
  {
    double y_hat = 0.0;
    int max_tap = i < n_taps - 1 ? i : n_taps - 1;

    for (int tap = 0; tap <= max_tap; tap++)
    {
      y_hat += h[tap] * x[i - tap];
    }

    res[i] = y_hat - y[i];
    data_loss += res[i] * res[i];
  }

  reg_loss = 0.0;
  if (grad != 0)
  {
    for (int i = 0; i < n; i++)
    {
      grad[i] = 0.0;
      reg_loss += z[i] * z[i];
    }

    for (int tap = 0; tap < n_taps; tap++)
    {
      for (int i = 0; i + tap < n; i++)
      {
        grad[i] += 2.0 * res[i + tap] * h[tap];
      }
    }

    for (int i = 0; i < n; i++)
    {
      double sech2 = 1.0 - x[i] * x[i];
      grad[i] = grad[i] * sech2 + 2.0 * gamma * z[i];
    }
  }
  else
  {
    for (int i = 0; i < n; i++)
    {
      reg_loss += z[i] * z[i];
    }
  }

  parity_loss = parity_loss_and_grad(info, x, lambda, grad, prefix, suffix);

  if (data_loss_out != 0)
  {
    *data_loss_out = data_loss;
  }
  if (parity_loss_out != 0)
  {
    *parity_loss_out = parity_loss;
  }

  return data_loss + lambda * parity_loss + gamma * reg_loss;
}

static double evaluate_full_objective(const parity_info *info, const double *y,
                                      const double *h, const double *h_prior,
                                      const double *z, int n, int n_taps,
                                      double lambda, double gamma, double eta,
                                      double *x, double *res,
                                      double *prefix, double *suffix,
                                      double *data_loss_out,
                                      double *parity_loss_out)
{
  double objective;
  double h_reg = 0.0;
  double h_prior_loss = 0.0;

  objective = evaluate_z_objective(info, y, h, z, n, n_taps, lambda, gamma,
                                   x, res, 0, prefix, suffix,
                                   data_loss_out, parity_loss_out);

  for (int i = 0; i < n_taps; i++)
  {
    double dh = h[i] - h_prior[i];
    h_reg += h[i] * h[i];
    h_prior_loss += dh * dh;
  }

  return objective + gamma * h_reg + eta * h_prior_loss;
}

static void optimize_z(const parity_info *info, const double *y, const double *h,
                       int n, int n_taps, const options *opts,
                       double *z, joint_workspace *ws)
{
  for (int step_idx = 0; step_idx < opts->z_steps; step_idx++)
  {
    double current;
    double grad_norm2;
    double step;
    int accepted;

    current = evaluate_z_objective(info, y, h, z, n, n_taps,
                                   opts->lambda, opts->gamma,
                                   ws->x, ws->res, ws->grad,
                                   ws->prefix, ws->suffix, 0, 0);

    grad_norm2 = 0.0;
    for (int i = 0; i < n; i++)
    {
      grad_norm2 += ws->grad[i] * ws->grad[i];
    }

    if (grad_norm2 <= 1e-10 * n)
    {
      break;
    }

    step = 1.0;
    accepted = 0;

    while (step >= MIN_STEP_SIZE)
    {
      double trial;

      for (int i = 0; i < n; i++)
      {
        ws->z_trial[i] = clamp_double(z[i] - step * ws->grad[i],
                                      -opts->z_limit, opts->z_limit);
      }

      trial = evaluate_z_objective(info, y, h, ws->z_trial, n, n_taps,
                                   opts->lambda, opts->gamma,
                                   ws->x, ws->res, 0,
                                   ws->prefix, ws->suffix, 0, 0);

      if (trial <= current - ARMIJO_C1 * step * grad_norm2)
      {
        memcpy(z, ws->z_trial, n * sizeof *z);
        accepted = 1;
        break;
      }

      step *= 0.5;
    }

    if (!accepted)
    {
      break;
    }
  }
}

static void hard_decision_from_z(const double *z, int n, char *bits)
{
  for (int i = 0; i < n; i++)
  {
    bits[i] = z[i] >= 0.0;
  }
}

static double restart_scale(int restart_idx)
{
  double scale = 1.0;

  for (int i = 0; i < restart_idx; i++)
  {
    scale *= 0.5;
  }

  return scale;
}

static void run_joint_decoder(mod2sparse *matrix, const parity_info *info,
                              const options *opts, const double *y,
                              const double *h_prior, joint_workspace *ws,
                              double *lratio, char *dblk, char *pchk,
                              double *bitpr, block_stats *stats)
{
  double total_start;
  double bp_start;
  double bp_end;
  double total_end;
  double best_objective;
  int best_direct_valid;
  int best_restart;
  double data_loss;
  double parity_loss;

  total_start = now_seconds();

  dfe_warm_start(y, ws->n, h_prior, ws->n_taps, opts->warm_clip,
                 ws->x, ws->res, ws->z0);

  best_objective = DBL_MAX;
  best_direct_valid = 0;
  best_restart = 0;

  for (int restart = 0; restart < opts->restarts; restart++)
  {
    double scale = restart_scale(restart);
    double objective;
    int direct_valid;

    for (int i = 0; i < ws->n; i++)
    {
      ws->z[i] = clamp_double(scale * ws->z0[i], -opts->z_limit, opts->z_limit);
    }
    memcpy(ws->h, h_prior, ws->n_taps * sizeof *ws->h);

    for (int outer = 0; outer < opts->alt_iters; outer++)
    {
      for (int i = 0; i < ws->n; i++)
      {
        ws->x[i] = tanh(ws->z[i]);
      }

      solve_supported_channel(ws->x, y, ws->n, h_prior, ws->n_taps,
                              opts->gamma, opts->eta,
                              ws->gram, ws->rhs, ws->h);

      optimize_z(info, y, ws->h, ws->n, ws->n_taps, opts, ws->z, ws);
    }

    objective = evaluate_full_objective(info, y, ws->h, h_prior, ws->z,
                                        ws->n, ws->n_taps,
                                        opts->lambda, opts->gamma, opts->eta,
                                        ws->x, ws->res,
                                        ws->prefix, ws->suffix, 0, 0);
    if (!(objective < DBL_MAX))
    {
      objective = DBL_MAX;
    }

    hard_decision_from_z(ws->z, ws->n, ws->direct_bits);
    direct_valid = check(matrix, ws->direct_bits, ws->direct_pchk) == 0;

    if (restart == 0
     || (direct_valid && !best_direct_valid)
     || (direct_valid == best_direct_valid && objective < best_objective))
    {
      best_objective = objective;
      best_direct_valid = direct_valid;
      best_restart = restart;
      memcpy(ws->best_z, ws->z, ws->n * sizeof *ws->best_z);
      memcpy(ws->best_h, ws->h, ws->n_taps * sizeof *ws->best_h);
    }
  }

  for (int i = 0; i < ws->n; i++)
  {
    lratio[i] = safe_exp(2.0 * ws->best_z[i]);
  }

  bp_start = now_seconds();
  max_iter = opts->bp_iters;
  stats->bp_iters = prprp_decode(matrix, lratio, dblk, pchk, bitpr);
  bp_end = now_seconds();
  stats->bp_valid = check(matrix, dblk, pchk) == 0;
  stats->objective = evaluate_full_objective(info, y, ws->best_h, h_prior, ws->best_z,
                                             ws->n, ws->n_taps,
                                             opts->lambda, opts->gamma, opts->eta,
                                             ws->x, ws->res,
                                             ws->prefix, ws->suffix,
                                             &data_loss, &parity_loss);
  total_end = now_seconds();
  stats->data_loss = data_loss;
  stats->parity_loss = parity_loss;
  stats->restart_idx = best_restart;
  stats->direct_valid = best_direct_valid;
  stats->bp_sec = bp_end - bp_start;
  stats->total_sec = total_end - total_start;
  stats->joint_sec = stats->total_sec - stats->bp_sec;
}
