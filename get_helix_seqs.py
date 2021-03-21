#! usr/bin/python

import os, sys
import numpy as np
import time as t
from MDAnalysis import *
import Bio.SeqIO

path = "example_input"

def parsePDB_toGetHelix(pdb,list):
        sublist = []
        helix_list = []
        sublist_index = -1
        index=0
        len_list = len(list)
        while index < len_list:
                next_index = index+1
                if list[index].split()[0]=="HELIX":
                        sublist.append(list[index].split())
                        sublist_index+=1
                        while sublist[sublist_index][8]==list[next_index].split()[5]:
                                sublist[sublist_index][8]=list[next_index].split()[8]
                                sublist[sublist_index][6]=list[next_index].split()[6]
                                del list[next_index]
                                len_list = len_list-1
                index=index+1
        count_helix = len(sublist)
        Amino_set = set(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'])
        for index in range(count_helix):
                try:
                        helix = universe.__getattribute__(sublist[index][4]).select_atoms("resid "+sublist[index][5]+":"+sublist[index][8]+" and not (name H*)")
                except:
                        helix = universe.select_atoms("segid "+sublist[index][4]+" and (resid "+sublist[index][5]+":"+sublist[index][8]+" and not (name H*))")
                helix_res_set = set(helix.resnames)
                if(len(helix_res_set-Amino_set)>0):
                        continue
                if abs(int(sublist[index][8])-int(sublist[index][5]))>5:
                        seq_record = helix.residues.sequence(id=pdb+"_"+sublist[index][4]+"_"+sublist[index][9]+"_"+sublist[index][5]+"_"+sublist[index][8], name="myprotein", description="")
                        seq_list.append(seq_record)

#Main function
start = t.time()
excluded_pdb_count=0
seq_list = []
count_pdb = 0

fr_trans = open("trans_mem_pro.txt", "r")
trans_pro_list = []
for line in fr_trans:
        if line[:4] not in trans_pro_list:
                trans_pro_list.append(line[:4])
fr_trans.close()

for file in os.listdir(path)[0:]:
        if file in trans_pro_list:
                continue
        count_pdb+=1
        try:
                start_per_pdb = t.time()
                list=[]
                current_file = os.path.join(path, file)
                fp = open(current_file, "r")
                PDB_name = file[0:4]
                print(str(count_pdb)+". "+ PDB_name)
                list = fp.readlines()
                fp.close()
                universe = Universe(current_file, permissive=False)
                parsePDB_toGetHelix(PDB_name,list)
                print("Execution time for "+PDB_name+":")
                print(t.time()-start_per_pdb)
                print("Overall execution time till here:")
                print(t.time()-start)

        except:
                raise
                excluded_pdb_count=excluded_pdb_count+1
                f_excl_pdb = open("seq_weird_pdbs.txt","a")
                f_excl_pdb.write(file[0:4])
                f_excl_pdb.write("\n")
                f_excl_pdb.close()
                continue

print(excluded_pdb_count)
Bio.SeqIO.write(seq_list, "sequences_all.fasta", "fasta")
print(t.time()-start)
