#!/bin/bash

DEST=jerc
mkdir -p "$DEST"

TEMP=tmp_jec

jec_folders=(
  Summer22_22Sep2023_V3_MC
  Summer22EE_22Sep2023_V3_MC
  Summer23Prompt23_V3_MC
  Summer23BPixPrompt23_V3_MC
  Summer24Prompt24_V1_MC
  Summer22_22Sep2023_RunCD_V3_DATA
  Summer22EE_22Sep2023_RunE_V3_DATA
  Summer22EE_22Sep2023_RunF_V3_DATA
  Summer22EE_22Sep2023_RunG_V3_DATA
  Summer23Prompt23_RunCv123_V3_DATA
  Summer23Prompt23_RunCv4_V3_DATA
  Summer23BPixPrompt23_RunD_V3_DATA
  Summer24Prompt24_RunCnib1_V1_DATA
  Summer24Prompt24_RunDnib1_V1_DATA
  Summer24Prompt24_RunEnib1_V1_DATA
  Summer24Prompt24_RunFnib1_V1_DATA
  Summer24Prompt24_RunGnib1_V1_DATA
  Summer24Prompt24_RunHnib1_V1_DATA
  Summer24Prompt24_RunInib1_V1_DATA
)

jer_folders=(
  Summer22_22Sep2023_JRV1_MC
  Summer22EE_22Sep2023_JRV1_MC
  Summer23Prompt23_RunCv1234_JRV1_MC
  Summer23BPixPrompt23_RunD_JRV1_MC
  Summer23BPixPrompt23_RunD_JRV1_MC
)

#JEC DATABASE
git clone --depth 1 https://github.com/cms-jet/JECDatabase.git "$TEMP"

for folder in "${jec_folders[@]}"; do
  src="$TEMP/textFiles/$folder"
  dest="$DEST/$folder"
  if [ -d "$src" ]; then
    mkdir -p "$dest"
    rsync -aL --include='*/' --include='*AK4PFPuppi.txt' --include='*AK8PFPuppi.txt' --exclude='*' "$src/" "$dest/"
  else
    echo "WARNING: Folder $folder not found!"
  fi
done

rm -rf "$TEMP"

#JER DATABASE
git clone --depth 1 https://github.com/cms-jet/JRDatabase.git "$TEMP"

for folder in "${jer_folders[@]}"; do
  src="$TEMP/textFiles/$folder"
  dest="$DEST/$folder"
  if [ -d "$src" ]; then
    mkdir -p "$dest"
    rsync -aL --include='*/' --include='*AK4PFPuppi.txt' --include='*AK8PFPuppi.txt' --exclude='*' "$src/" "$dest/"
  else
    echo "WARNING: Folder $folder not found!"
  fi
done

rm -rf "$TEMP"


#fix filenames for coffea extractor expectation
find "$DEST" -type f -name "*Uncertainty*.txt" ! -name "*.junc.txt" | while read file; do
    new_name="${file%.txt}.junc.txt"
    mv "$file" "$new_name"
done

find "$DEST" -type f -name "*_PtResolution_*.txt" ! -name "*.jr.txt" | while read file; do
    new_name="${file%.txt}.jr.txt"
    mv "$file" "$new_name"
done

find "$DEST" -type f -name "*_SF_*.txt" ! -name "*.jersf.txt" | while read file; do
    new_name="${file%.txt}.jersf.txt"
    mv "$file" "$new_name"
done

#fix for coffea extractor 
#which jetmet hasn't changed even though they changed the length of their jr filenames
for folder in "$DEST"/*_JR*; do
    folder_name=$(basename "$folder")
    num_parts=$(echo "$folder_name" | awk -F"_" '{print NF}')
    if [ "$num_parts" -eq 4 ]; then
        new_folder_name="$(echo "$folder_name" | sed 's/_//')"
        mv "$folder" "$DEST/$new_folder_name"\

        find "$DEST/$new_folder_name" -type f | while read f; do
            base=$(basename "$f")
            dir=$(dirname "$f")

            new_file_name="$(echo "$base" | sed 's/_//')"
            mv "$f" "$dir/$new_file_name"
        done
    fi
done

find "$DEST" -type f -name "*.txt" -exec gzip -f {} \;