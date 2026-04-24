/* EQUALIZE_DFE.C - Decision-feedback equalizer for short ISI channels. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "open.h"

void usage(void);

static void equalize_block(FILE *out, const double *rx, int n_bits,
                           const double *taps, int n_taps)
{
  int i, j;
  double *decisions;

  decisions = malloc(n_bits * sizeof *decisions);
  if (decisions == 0)
  {
    fprintf(stderr, "Out of memory\n");
    exit(1);
  }

  for (i = 0; i < n_bits; i++)
  {
    double fb = 0.0;
    double eq;

    for (j = 1; j < n_taps; j++)
    {
      int idx = i - j;
      if (idx >= 0)
      {
        fb += taps[j] * decisions[idx];
      }
    }

    eq = (rx[i] - fb) / taps[0];
    decisions[i] = eq >= 0.0 ? 1.0 : -1.0;
    fprintf(out, " %+5.2f", eq);
  }

  putc('\n', out);
  free(decisions);
}

int main(int argc, char **argv)
{
  char *rx_file, *eq_file;
  FILE *in, *out;
  char junk;
  int block_size;
  int n_taps;
  int block_count;
  double *taps;
  double *rx;

  if (!(rx_file = argv[1])
   || !(eq_file = argv[2])
   || !argv[3] || sscanf(argv[3], "%d%c", &block_size, &junk) != 1 || block_size <= 0
   || argc < 5)
  {
    usage();
  }

  n_taps = argc - 4;
  taps = malloc(n_taps * sizeof *taps);
  rx = malloc(block_size * sizeof *rx);
  if (taps == 0 || rx == 0)
  {
    fprintf(stderr, "Out of memory\n");
    exit(1);
  }

  for (int i = 0; i < n_taps; i++)
  {
    if (sscanf(argv[4+i], "%lf%c", &taps[i], &junk) != 1)
    {
      usage();
    }
  }

  if (fabs(taps[0]) < 1e-12)
  {
    fprintf(stderr, "First tap must be non-zero for DFE\n");
    exit(1);
  }

  in = open_file_std(rx_file, "r");
  if (in == 0)
  {
    fprintf(stderr, "Can't open received file: %s\n", rx_file);
    exit(1);
  }

  out = open_file_std(eq_file, "w");
  if (out == 0)
  {
    fprintf(stderr, "Can't create equalized file: %s\n", eq_file);
    exit(1);
  }

  block_count = 0;
  for (;;)
  {
    int i;
    for (i = 0; i < block_size; i++)
    {
      if (fscanf(in, "%lf", &rx[i]) != 1)
      {
        break;
      }
    }

    if (i == 0)
    {
      break;
    }
    if (i != block_size)
    {
      fprintf(stderr, "Short received block at end of %s\n", rx_file);
      exit(1);
    }

    equalize_block(out, rx, block_size, taps, n_taps);
    block_count += 1;
  }

  fprintf(stderr, "Equalized %d blocks with DFE\n", block_count);

  if (ferror(out) || fclose(out) != 0)
  {
    fprintf(stderr, "Error writing equalized data to %s\n", eq_file);
    exit(1);
  }

  fclose(in);
  free(taps);
  free(rx);
  return 0;
}

void usage(void)
{
  fprintf(stderr,
    "Usage: equalize_dfe received-file equalized-file block-size tap1 [tap2 ...]\n");
  exit(1);
}
