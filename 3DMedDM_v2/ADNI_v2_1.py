#T1/T2/FLAIR/PD/tau  ADNI1
import os
import shutil
count = 0
def copy_first_unique_file(source_folder, destination_folder):
    for files in os.listdir(source_folder):
        for file in os.listdir(os.path.join(source_folder, files)):
            # if "PIB" in file:  #PET_PIB # tried, and no resultsex
            if "FDG" in file or "PET" in file or "ADNI" in file or "PET_BRAIN" in file and "PIB" not in file and "AV45" not in file:  #PET_FDG
            # if "MPRAGE" in file or "MP-RAGE" in file and "REPEAT" not in file and "REPEAT" not in file and "MPRAGE_" not in file:   #MRI_T1
                for j in sorted(os.listdir(os.path.join(source_folder, files, file))):
                    for k in sorted(os.listdir(os.path.join(source_folder, files, file, j))):
                        filename_year = j.split('-')[0] + '-' + k
                        dest_subfolder = os.path.join(destination_folder, files)
                        src_file_path = os.path.join(source_folder, files, file, j, k)
                        shutil.move(src_file_path, os.path.join(dest_subfolder, filename_year))

source = '/backup/ADNI1/ADNI/'
destination = '/public_bme/data/ADNI1/ADNI1_PET/ADNI1_PET/'
if not os.path.exists(destination):
    os.makedirs(destination)
copy_first_unique_file(source, destination)
