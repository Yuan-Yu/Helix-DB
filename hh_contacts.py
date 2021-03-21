#! usr/bin/python

import os, sys
import numpy as np
import time as t
from MDAnalysis import *
from numba import jit
import pickle

@jit('void(f4[:,:],f4[:,:],f4,i4[:,:])',nopython=True)
def pairwiseatomContact(crdA,crdB,cutOff,contactMap):
    cutOff_sq = cutOff**2
    lenA = crdA.shape[0]
    lenB = crdB.shape[0]

    for indexA in range(lenA):
        for indexB in range(lenB):
            sum_diff_sqr=0
            x_diff=crdA[indexA, 0] - crdB[indexB, 0]
            if x_diff>cutOff:
                contactMap[indexA,indexB]=0
                continue
            sum_diff_sqr += x_diff ** 2

            y_diff=crdA[indexA, 1] - crdB[indexB, 1]
            if y_diff>cutOff:
                contactMap[indexA,indexB]=0
                continue
            sum_diff_sqr +=  y_diff** 2
            if sum_diff_sqr>cutOff_sq:
                contactMap[indexA,indexB]=0
                continue

            z_diff =crdA[indexA, 2] - crdB[indexB, 2]
            if z_diff>cutOff:
                contactMap[indexA,indexB]=0
                continue
            sum_diff_sqr += z_diff ** 2
            if sum_diff_sqr<cutOff_sq:
                contactMap[indexA,indexB]=1
            else:
                contactMap[indexA,indexB]=0

def pairwiseResContact(atomicContactMap,numOfAtomsListA,numOfAtomsListB,residueListA,residueListB,res_level_mat):
        num_residue_A = len(numOfAtomsListA)
        num_residue_B = len(numOfAtomsListB)
        pre_atoms_sumA = 0
        pre_atoms_sumB = 0

        for indexA in range(num_residue_A):
                if indexA!=0:
                        pre_atoms_sumA = pre_atoms_sumA + numOfAtomsListA[indexA-1]
                for indexB in range(num_residue_B):
                        if indexB!=0:
                                pre_atoms_sumB = pre_atoms_sumB + numOfAtomsListB[indexB-1]
                        elif indexB==0:
                                pre_atoms_sumB = 0
                        res_level_mat[indexA, indexB] = atomicContactMap[pre_atoms_sumA:pre_atoms_sumA+numOfAtomsListA[indexA], pre_atoms_sumB:pre_atoms_sumB+numOfAtomsListB[indexB]].sum()
def parsePDB_toGetHelix(list_coord_helix, list_coord_backbone, list_coord_sidechain, bb_num_atom_list, ss_num_atom_list, list_residue_bb, list_residue_ss, helix_chain, helix_id, len_list, list, start_resid, end_resid):
        sublist = []
        helix_list = []
        sublist_index = -1
        index=0
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
        count_helix_old = len(sublist)
        count_helix = count_helix_old
        Amino_set = set(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'])
        for index in range(count_helix_old):
                try:
                        helix = universe.__getattribute__(sublist[index][4]).select_atoms("resid "+sublist[index][5]+":"+sublist[index][8]+" and not (name H*)")
                except:
                        helix = universe.select_atoms("segid "+sublist[index][4]+" and (resid "+sublist[index][5]+":"+sublist[index][8]+" and not (name H*))")
                helix_res_set = set(helix.resnames)
                if(len(helix_res_set-Amino_set)>0):
                        count_helix = count_helix-1
                        continue

                helix_chain.append(sublist[index][4])
                helix_id.append(sublist[index][9])
                start_resid.append(sublist[index][5])
                end_resid.append(sublist[index][8])

                list_coord_helix.append(helix.positions)

                backbone = helix.select_atoms("backbone")
                bb_num_atom_sublist = []
                list_residue_bb_sublist = []
                for residue in backbone.split("residue"):
                        bb_num_atom_sublist.append(residue.n_atoms)
                for residue in backbone.residues:
                        list_residue_bb_sublist.append(residue)
                bb_num_atom_list.append(bb_num_atom_sublist)
                list_residue_bb.append(list_residue_bb_sublist)
                list_coord_backbone.append(backbone.positions)

                sidechain = helix.select_atoms("not backbone or (resname GLY and name CA)")
                ss_num_atom_sublist = []
                list_residue_ss_sublist = []
                for residue in sidechain.split("residue"):
                        ss_num_atom_sublist.append(residue.n_atoms)
                for residue in sidechain.residues:
                        list_residue_ss_sublist.append(residue)
                ss_num_atom_list.append(ss_num_atom_sublist)
                list_residue_ss.append(list_residue_ss_sublist)
                list_coord_sidechain.append(sidechain.positions)

        return count_helix

def MaxResContact(res_level_matrix,residueListA,residueListB,Helix_A,Helix_B,Amino_to_Num,res_max_contact):
        for indexA in range(len(residueListA)):
                num_contactsA = 0
                for indexB in range(len(residueListB)):
                        num_contactsA+=res_level_matrix[indexA, indexB]
                if len(res_max_contact[Amino_to_Num[residueListA[indexA][0:3]]])==0:
                        res_max_contact[Amino_to_Num[residueListA[indexA][0:3]]].append(num_contactsA)
                        res_max_contact[Amino_to_Num[residueListA[indexA][0:3]]].append(Helix_A)
                elif len(res_max_contact[Amino_to_Num[residueListA[indexA][0:3]]])!=0 and num_contactsA>=res_max_contact[Amino_to_Num[residueListA[indexA][0:3]]][0]:
                        del res_max_contact[Amino_to_Num[residueListA[indexA][0:3]]][0:2]
                        res_max_contact[Amino_to_Num[residueListA[indexA][0:3]]].append(num_contactsA)
                        res_max_contact[Amino_to_Num[residueListA[indexA][0:3]]].append(Helix_A)
        for indexB in range(len(residueListB)):
                num_contactsB = 0
                for indexA in range(len(residueListA)):
                        num_contactsB+=res_level_matrix[indexA, indexB]
                if len(res_max_contact[Amino_to_Num[residueListB[indexB][0:3]]])==0:
                        res_max_contact[Amino_to_Num[residueListB[indexB][0:3]]].append(num_contactsB)
                        res_max_contact[Amino_to_Num[residueListB[indexB][0:3]]].append(Helix_B)
                elif len(res_max_contact[Amino_to_Num[residueListB[indexB][0:3]]])!=0 and num_contactsB>=res_max_contact[Amino_to_Num[residueListB[indexB][0:3]]][0]:
                        del res_max_contact[Amino_to_Num[residueListB[indexB][0:3]]][0:2]
                        res_max_contact[Amino_to_Num[residueListB[indexB][0:3]]].append(num_contactsB)
                        res_max_contact[Amino_to_Num[residueListB[indexB][0:3]]].append(Helix_B)


#Main function
start = t.time()
count_pdb = 0
path = "example_input"
Dictionary = {}
amino_matrix = np.zeros([20, 20], dtype = "int32")
excluded_pdb_count=0
res_max_contact=[[]for i in range(20)]
Amino_to_Num = {'ALA':0,'ARG':1,'ASN':2,'ASP':3,'CYS':4,'GLU':5,'GLN':6,'GLY':7,'HIS':8,'ILE':9,'LEU':10,'LYS':11,'MET':12,'PHE':13,'PRO':14,'SER':15,'THR':16,'TRP':17,'TYR':18,'VAL':19}
#Class
class res_lev_mat_info:
        def __init__(self, rList_row, rList_col, helix_row, helix_col):
                self.row = len(rList_row)
                self.col = len(rList_col)
                self.matrix_bb = np.zeros([self.row, self.col], dtype = "int32")
                self.matrix_bs = np.zeros([self.row, self.col], dtype = "int32")
                self.matrix_sb = np.zeros([self.row, self.col], dtype = "int32")
                self.matrix_ss = np.zeros([self.row, self.col], dtype = "int32")
                self.rList_row = rList_row
                self.rList_col = rList_col
                self.helix_row = helix_row
                self.helix_col = helix_col
        def get_NOTbool_matrix_hh(self):
                mat_bb = self.matrix_bb.copy()
                mat_bs = self.matrix_bs.copy()
                mat_sb = self.matrix_sb.copy()
                mat_ss = self.matrix_ss.copy()
                NOTbool_matrix_hh = mat_bb+mat_bs+mat_sb+mat_ss
                return NOTbool_matrix_hh
        def get_bool_matrix_hh(self):
                bool_matrix_hh = self.get_NOTbool_matrix_hh()
                non_zero_index = np.nonzero(bool_matrix_hh)
                for index in range(len(non_zero_index[0])):
                        bool_matrix_hh[non_zero_index[0][index],non_zero_index[1][index]]=1
                return bool_matrix_hh
        def get_bool_matrix_bb(self):
                bool_matrix_bb = self.matrix_bb.copy()
                non_zero_index_bb = np.nonzero(bool_matrix_bb)
                for index in range(len(non_zero_index_bb[0])):
                        bool_matrix_bb[non_zero_index_bb[0][index],non_zero_index_bb[1][index]]=1
                return bool_matrix_bb
        def get_bool_matrix_bs(self):
                bool_matrix_bs = self.matrix_bs.copy()
                non_zero_index_bs = np.nonzero(bool_matrix_bs)
                for index in range(len(non_zero_index_bs[0])):
                        bool_matrix_bs[non_zero_index_bs[0][index],non_zero_index_bs[1][index]]=1
                return bool_matrix_bs
        def get_bool_matrix_sb(self):
                bool_matrix_sb = self.matrix_sb.copy()
                non_zero_index_sb = np.nonzero(bool_matrix_sb)
                for index in range(len(non_zero_index_sb[0])):
                        bool_matrix_sb[non_zero_index_sb[0][index],non_zero_index_sb[1][index]]=1
                return bool_matrix_sb
        def get_bool_matrix_ss(self):
                bool_matrix_ss = self.matrix_ss.copy()
                non_zero_index_ss = np.nonzero(bool_matrix_ss)
                for index in range(len(non_zero_index_ss[0])):
                        bool_matrix_ss[non_zero_index_ss[0][index],non_zero_index_ss[1][index]]=1
                return bool_matrix_ss
        def get_amino_matrix_hh(self):
                amino_matrix_hh = np.zeros([20, 20], dtype = "int32")
                bool_matrix_hh  = self.get_bool_matrix_hh()
                for indexA in range(len(self.rList_row)):
                        for indexB in range(len(self.rList_col)):
                                if bool_matrix_hh[indexA, indexB]==1:
                                        amino_matrix_hh[Amino_to_Num[self.rList_row[indexA][0:3]], Amino_to_Num[self.rList_col[indexB][0:3]]]+=1
                                        amino_matrix_hh[Amino_to_Num[self.rList_col[indexB][0:3]], Amino_to_Num[self.rList_row[indexA][0:3]]]+=1
                return amino_matrix_hh
        def get_resPair_list(self):
                NOTbool_matrix_hh = self.get_NOTbool_matrix_hh()
                resPairList=[]
                for indexA in range(len(self.rList_row)):
                        for indexB in range(len(self.rList_col)):
                                if(NOTbool_matrix_hh[indexA, indexB]>0):
                                        resPairList_sub=[]
                                        resPairList_sub.append(self.rList_row[indexA])
                                        resPairList_sub.append(self.rList_col[indexB])
                                        resPairList_sub.append(NOTbool_matrix_hh[indexA, indexB])
                                        resPairList.append(resPairList_sub)
                return resPairList


fw_hh = open(f"helix_contact_all.txt", "a")

fr_trans = open("trans_mem_pro.txt", "r")
trans_pro_list = []
for line in fr_trans:
        if line[:4] not in trans_pro_list:
                trans_pro_list.append(line[:4])

fw_hh_dict = open(f"hh_dictionary_all.txt", "a")
for file in os.listdir(path)[:]:
  if file in trans_pro_list:
        continue
  try:
        start_per_pdb = t.time()
        dict_hh = {}
        count_pdb+=1
        sum_backbone=0; sum_bb_sidechain=0; sum_sidechain_bb=0; sum_ss=0
        list=[]
        current_file = os.path.join(path, file)
        fp = open(current_file, "r")
        print(str(count_pdb)+". "+file[0:4])
        PDB_name = file[0:4]
        list = fp.readlines()
        len_list = len(list)

        fp.close()

        universe = Universe(current_file, permissive=False)

        list_coord_helix = []; list_coord_backbone = []; list_coord_sidechain = []
        bb_num_atom_list = []; ss_num_atom_list = []
        list_residue_bb = []; list_residue_ss = []
        helix_chain=[]; helix_id = [];
        start_resid=[]; end_resid=[]

        count_helix = parsePDB_toGetHelix(list_coord_helix, list_coord_backbone, list_coord_sidechain, bb_num_atom_list, ss_num_atom_list, list_residue_bb, list_residue_ss, helix_chain, helix_id, len_list, list, start_resid, end_resid)
        fw_hh.write("PDB "+PDB_name+"\n")

        Helix_1="";Helix_2="";
        for k in range(count_helix):
                helix_1 = []; Helix_1="";
                for l in range(k+1, count_helix):
                        helix_2 = []; Helix_2=""

                        pairwiseMatrix_helix = np.empty([len(list_coord_helix[k]), len(list_coord_helix[l])], dtype = "int32")
                        pairwiseatomContact(list_coord_helix[k], list_coord_helix[l], 4, pairwiseMatrix_helix)
                        num_helix_interaction = 0
                        sum_backbone_hh=0; sum_bb_sidechain_hh=0; sum_sidechain_bb_hh=0; sum_ss_hh=0

                        rList_row=[];rList_col=[]
                        if pairwiseMatrix_helix.any()==True:
                                if ((helix_chain[k]==helix_chain[l] and abs(int(end_resid[k])-int(start_resid[l]))>10 and abs(int(end_resid[l])-int(start_resid[k]))>10) or helix_chain[k]!=helix_chain[l]):
                                        Helix_1 = PDB_name+"_"+helix_chain[k]+"_"+helix_id[k]+"_"+start_resid[k]+"_"+end_resid[k]
                                        Helix_2 = PDB_name+"_"+helix_chain[l]+"_"+helix_id[l]+"_"+start_resid[l]+"_"+end_resid[l]
                                        for indexRow in range(len(list_residue_bb[k])):
                                                rList_row.append(str(list_residue_bb[k][indexRow])[9:12]+"_"+str(list_residue_bb[k][indexRow]).strip(">")[14:])
                                        for indexCol in range(len(list_residue_bb[l])):
                                                rList_col.append(str(list_residue_bb[l][indexCol])[9:12]+"_"+str(list_residue_bb[l][indexCol]).strip(">")[14:])

                                        res_level_mat = res_lev_mat_info(rList_row, rList_col, Helix_1, Helix_2)

                                        pairwiseMatrix_backbone = np.empty([len(list_coord_backbone[k]), len(list_coord_backbone[l])], dtype = "int32")
                                        pairwiseatomContact(list_coord_backbone[k], list_coord_backbone[l], 4, pairwiseMatrix_backbone)
                                        pairwiseResContact(pairwiseMatrix_backbone,bb_num_atom_list[k],bb_num_atom_list[l],list_residue_bb[k], list_residue_bb[l],res_level_mat.matrix_bb)
                                        bool_matrix_bb = res_level_mat.get_bool_matrix_bb()
                                        sum_backbone_hh = bool_matrix_bb.sum()
                                        sum_backbone+=sum_backbone_hh

                                        pairwiseMatrix_bb_sidechain = np.empty([len(list_coord_backbone[k]), len(list_coord_sidechain[l])], dtype = "int32")
                                        pairwiseatomContact(list_coord_backbone[k], list_coord_sidechain[l], 4, pairwiseMatrix_bb_sidechain)
                                        pairwiseResContact(pairwiseMatrix_bb_sidechain,bb_num_atom_list[k],ss_num_atom_list[l],list_residue_bb[k], list_residue_ss[l],res_level_mat.matrix_bs)

                                        list_len = len(list_residue_ss[l])
                                        for gly in range(list_len):
                                                if str(list_residue_ss[l][gly])[9:12]=="GLY":
                                                        res_level_mat.matrix_bs[:,gly]=0
                                        bool_matrix_bs = res_level_mat.get_bool_matrix_bs()
                                        sum_bb_sidechain_hh  = bool_matrix_bs.sum()
                                        sum_bb_sidechain +=sum_bb_sidechain_hh

                                        pairwiseMatrix_sidechain_bb = np.empty([len(list_coord_sidechain[k]), len(list_coord_backbone[l])], dtype = "int32")
                                        pairwiseatomContact(list_coord_sidechain[k], list_coord_backbone[l], 4, pairwiseMatrix_sidechain_bb)
                                        pairwiseResContact(pairwiseMatrix_sidechain_bb,ss_num_atom_list[k],bb_num_atom_list[l],list_residue_ss[k],list_residue_bb[l],res_level_mat.matrix_sb)

                                        list_len = len(list_residue_ss[k])
                                        for gly in range(list_len):
                                                if str(list_residue_ss[k][gly])[9:12]=="GLY":
                                                        res_level_mat.matrix_sb[gly,:]=0
                                        bool_matrix_sb = res_level_mat.get_bool_matrix_sb()
                                        sum_sidechain_bb_hh = bool_matrix_sb.sum()
                                        sum_sidechain_bb += sum_sidechain_bb_hh

                                        pairwiseMatrix_ss = np.empty([len(list_coord_sidechain[k]), len(list_coord_sidechain[l])], dtype = "int32")
                                        pairwiseatomContact(list_coord_sidechain[k], list_coord_sidechain[l], 4, pairwiseMatrix_ss)
                                        pairwiseResContact(pairwiseMatrix_ss,ss_num_atom_list[k],ss_num_atom_list[l],list_residue_ss[k],list_residue_ss[l],res_level_mat.matrix_ss)

                                        list_len = len(list_residue_ss[k])
                                        for gly in range(list_len):
                                                if str(list_residue_ss[k][gly])[9:12]=="GLY":
                                                        res_level_mat.matrix_ss[gly,:]=0
                                        list_len = len(list_residue_ss[l])
                                        for gly in range(list_len):
                                                if str(list_residue_ss[l][gly])[9:12]=="GLY":
                                                        res_level_mat.matrix_ss[:,gly]=0
                                        bool_matrix_ss = res_level_mat.get_bool_matrix_ss()
                                        sum_ss_hh = bool_matrix_ss.sum()
                                        sum_ss += sum_ss_hh

                                        bool_matrix_hh = res_level_mat.get_bool_matrix_hh()
                                        num_helix_interaction = bool_matrix_hh.sum()
                                        amino_matrix_hh = res_level_mat.get_amino_matrix_hh()

                                        MaxResContact(bool_matrix_hh,res_level_mat.rList_row,res_level_mat.rList_col,Helix_1,Helix_2,Amino_to_Num,res_max_contact)
                                        if (Helix_1+":"+Helix_2 in dict_hh)==False:
                                                dict_hh[Helix_1+":"+Helix_2]=res_level_mat
                                        amino_matrix+=amino_matrix_hh

                                        #Creating Dictionary
                                        helix_2.append(Helix_2)
                                        helix_2.append(str(sum_backbone_hh))
                                        helix_2.append(str(sum_bb_sidechain_hh+sum_sidechain_bb_hh ))
                                        helix_2.append(str(sum_ss_hh))
                                        helix_2.append(str(num_helix_interaction))
                                        helix_1.append(helix_2)
                                        if (Helix_1 in Dictionary)==False:
                                                Dictionary[Helix_1]=[]

                                        fw_hh.write("HH "+PDB_name+"_"+helix_chain[k]+"_"+helix_id[k]+"_"+start_resid[k]+"_"+end_resid[k]+" : "+PDB_name+"_"+helix_chain[l]+"_"+helix_id[l]+"_"+start_resid[l]+"_"+end_resid[l]+"\n")
                if (Helix_1 in Dictionary)==True:
                        Dictionary[Helix_1] = helix_1
                        fw_hh_dict.write(Helix_1+" : "+str(Dictionary[Helix_1])+"\n")#
                else:
                        continue
        fw_hh_dict.write("\n")
        fw_hh.write("\nBB bb "+str(sum_backbone)+"\nSB sb "+ str(sum_bb_sidechain+sum_sidechain_bb)+"\nSS ss "+ str(sum_ss)+"\n")
        fw_hh.write("\nI interactions "+str(sum_backbone+sum_bb_sidechain+sum_sidechain_bb+sum_ss)+"\nH helix "+str(count_helix)+"\nEND\n\n")

        f_dict = open('dictionaries/'+PDB_name,'wb')
        pickle.dump(dict_hh, f_dict, pickle.HIGHEST_PROTOCOL)
        f_dict.close()
        print("Execution time for "+PDB_name+":")
        print(t.time()-start_per_pdb)
        print("Overall execution time till here:")
        print(t.time()-start)
  except:
        raise
        excluded_pdb_count=excluded_pdb_count+1
        f_excl_pdb = open("weird_pdbs.txt","a")
        f_excl_pdb.write(file[0:4])
        f_excl_pdb.write("\n")
        f_excl_pdb.close()
        continue
fw_hh.write("\nIM "+str(amino_matrix)+"\n\nALA "+str(res_max_contact[0])+"\nARG "+str(res_max_contact[1])+"\nASN "+str(res_max_contact[2])+"\nASP "+str(res_max_contact[3])+"\nCYS "+str(res_max_contact[4])+"\nGLU "+str(res_max_contact[5])+"\nGLN "+str(res_max_contact[6])+"\nGLY "+str(res_max_contact[7])+"\nHIS "+str(res_max_contact[8])+"\nILE "+str(res_max_contact[9])+"\nLEU "+str(res_max_contact[10])+"\nLYS "+str(res_max_contact[11])+"\nMET "+str(res_max_contact[12])+"\nPHE "+str(res_max_contact[13])+"\nPRO "+str(res_max_contact[14])+"\nSER "+str(res_max_contact[15])+"\nTHR "+str(res_max_contact[16])+"\nTRP "+str(res_max_contact[17])+"\nTYR "+str(res_max_contact[18])+"\nVAL "+str(res_max_contact[19]))
fw_hh.close()
fw_hh_dict.close()
print(excluded_pdb_count)
print(t.time()-start)

