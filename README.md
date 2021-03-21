# Helix-DB
A python code for collecting helix information from PDB files.

## How
Clone this repository by 
```
git clone https://github.com/Yuan-Yu/Helix-DB
cd Helix-DB
```
Then,put the PDB files in **example_input** folder.  
  
1. For extracting the sequence of helixes from PDB files.  
```bash
python get_helix_seqs.py # The output file that contains the sequences of the helixes is "sequences_all.fasta"
```
2. For calculation of the contact information of helix from each PDB file.  
```bash
python hh_contacts.py 
# The output files are in the "dictionaries" folder.
# The log file is "hh_dictionary_all.txt"
```

