# Data Directory

Place your genomic FASTA files here.

## Expected Files

- `*.fasta` or `*.fa` - Genome sequence files
- Gzipped files (`.fasta.gz`) will need to be uncompressed first

## Download C. elegans Genome

```bash
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/002/985/GCF_000002985.6_WBcel235/GCF_000002985.6_WBcel235_genomic.fna.gz
gunzip GCF_000002985.6_WBcel235_genomic.fna.gz
mv GCF_000002985.6_WBcel235_genomic.fna celegans_genome.fasta