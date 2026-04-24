/* TRANSMIT_ISI.C - Simulate BPSK transmission through a short ISI channel. */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "open.h"
#include "rand.h"

void usage(void);

static void write_block(FILE *rf, const char *bits, int n_bits,
                        const double *taps, int n_taps, double std_dev)
{
  int i, j;

  for (i = 0; i < n_bits; i++)
  {
    double sample = 0.0;

    for (j = 0; j < n_taps; j++)
    {
      int idx = i - j;
      if (idx >= 0)
      {
        sample += taps[j] * (bits[idx]=='1' ? 1.0 : -1.0);
      }
    }

    sample += std_dev * rand_gaussian();
    fprintf(rf, " %+5.2f", sample);
  }

  putc('\n', rf);
}

int main(int argc, char **argv)
{
  char *tfile, *rfile;
  FILE *tf, *rf;
  char junk;
  int seed;
  int block_size, n_blocks;
  int n_bits_total;
  int bit_count;
  int n_taps;
  double std_dev;
  double *taps;

  if (!(tfile = argv[1])
   || !(rfile = argv[2])
   || !argv[3] || sscanf(argv[3], "%d%c", &seed, &junk)!=1
   || !argv[4] || sscanf(argv[4], "%lf%c", &std_dev, &junk)!=1
   || argc < 6)
  {
    usage();
  }

  n_taps = argc - 5;
  taps = malloc(n_taps * sizeof *taps);
  if (taps == 0)
  {
    fprintf(stderr, "Out of memory\n");
    exit(1);
  }

  for (int i = 0; i < n_taps; i++)
  {
    if (sscanf(argv[5+i], "%lf%c", &taps[i], &junk)!=1)
    {
      usage();
    }
  }

  tf = 0;
  block_size = 0;
  n_blocks = 0;

  if (sscanf(tfile, "%d%c", &n_bits_total, &junk)==1 && n_bits_total>0)
  {
    block_size = 1;
    n_blocks = n_bits_total;
  }
  else if (sscanf(tfile, "%dx%d%c", &block_size, &n_blocks, &junk)==2
            && block_size>0 && n_blocks>0)
  {
    n_bits_total = block_size * n_blocks;
  }
  else
  {
    tf = open_file_std(tfile, "r");
    if (tf == 0)
    {
      fprintf(stderr, "Can't open encoded file to transmit: %s\n", tfile);
      exit(1);
    }
  }

  rf = open_file_std(rfile, "w");
  if (rf == 0)
  {
    fprintf(stderr, "Can't create file for received data: %s\n", rfile);
    exit(1);
  }

  rand_seed(10*seed+3);
  bit_count = 0;

  if (tf == 0)
  {
    char *zero_block = malloc(block_size * sizeof *zero_block);
    if (zero_block == 0)
    {
      fprintf(stderr, "Out of memory\n");
      exit(1);
    }

    memset(zero_block, '0', block_size);

    for (int blk = 0; blk < n_blocks; blk++)
    {
      write_block(rf, zero_block, block_size, taps, n_taps, std_dev);
      bit_count += block_size;
    }

    free(zero_block);
  }
  else
  {
    char *line = 0;
    size_t line_cap = 0;
    ssize_t line_len;

    while ((line_len = getline(&line, &line_cap, tf)) != -1)
    {
      char *bits = malloc((line_len + 1) * sizeof *bits);
      int n_bits = 0;

      if (bits == 0)
      {
        fprintf(stderr, "Out of memory\n");
        exit(1);
      }

      for (ssize_t i = 0; i < line_len; i++)
      {
        char c = line[i];
        if (c == '0' || c == '1')
        {
          bits[n_bits++] = c;
        }
        else if (!isspace((unsigned char)c))
        {
          fprintf(stderr, "Bad character (code %d) in file being transmitted\n", c);
          exit(1);
        }
      }

      if (n_bits > 0)
      {
        write_block(rf, bits, n_bits, taps, n_taps, std_dev);
        bit_count += n_bits;
      }

      free(bits);
    }

    free(line);
    fclose(tf);
  }

  fprintf(stderr, "Transmitted %d bits through %d-tap ISI channel\n", bit_count, n_taps);

  if (ferror(rf) || fclose(rf)!=0)
  {
    fprintf(stderr, "Error writing received bits to %s\n", rfile);
    exit(1);
  }

  free(taps);
  return 0;
}

void usage(void)
{
  fprintf(stderr,
    "Usage: transmit_isi encoded-file|n-zeros received-file seed noise-std tap1 [tap2 ...]\n");
  exit(1);
}
