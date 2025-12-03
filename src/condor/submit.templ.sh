
#!/bin/bash

# remove old files
rm *.pkl
rm *.parquet

for t2_prefix in ${t2_prefixes}
do
    for folder in pickles parquet githashes
    do
        xrdfs $${t2_prefix} mkdir -p "/${outdir}/$${folder}"
    done
done

# clone repository
# try 3 times in case of network errors
(
    r=3
    # shallow clone of single branch (keep repo size as small as possible)
    while ! git clone --single-branch --branch $branch --depth=1 https://github.com/DAZSLE/hbb-run3.git
    do
        ((--r)) || exit
        sleep 60
    done
)
cd hbb-run3 || exit

commithash=$$(git rev-parse HEAD)
echo "https://github.com/DAZSLE/hbb-run3/commit/$${commithash}" > commithash.txt

pip install -e .

# run code 
python -u -W ignore $script --year $year --starti $starti --endi $endi --samples $sample --subsamples $subsample --nano-version ${nano_version} --${run_mode}

# Move final output to EOS
# This new logic recursively copies the region directories created by the processor

# --- FINAL COPY LOGIC ---
# This logic creates the nested structure and partN.parquet names

# 1. First, handle the githash and pickle files
xrdfs ${t2_prefixes} mkdir -p "/${outdir}/githashes"
xrdcp -f commithash.txt "${t2_prefixes}/${outdir}/githashes/commithash_${jobnum}.txt"

xrdfs ${t2_prefixes} mkdir -p "/${outdir}/pickles"
xrdcp -f *.pkl "${t2_prefixes}/${outdir}/pickles/out_${jobnum}.pkl"

# 2. Next, handle the combined parquet files
for file in *.parquet; do
    # Extract the region name from the local filename (e.g., gets "control-tt" from "control-tt.parquet")
    base_file=$$(basename "$${file}" ".parquet")
    region_name="$${base_file##*_}"
    jer_name="$${base_file%_*}"

    # Create the region-specific subdirectory on EOS
    xrdfs ${t2_prefixes} mkdir -p "/${outdir}/parquet/$${jer_name}/$${region_name}"

    # Define the final filename using the job number for uniqueness
    final_filename="part${jobnum}.parquet"

    # Copy the file to its final, nested destination with the new name
    xrdcp -f "$$file" "${t2_prefixes}/${outdir}/parquet/$${jer_name}/$${region_name}/$${final_filename}"
done



rm *.parquet
rm *.pkl
rm commithash.txt
