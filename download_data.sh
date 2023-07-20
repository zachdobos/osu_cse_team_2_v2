gsutil -m cp -r gs://i-care-2.0.physionet.org/training/*/*.txt .

for ((i=0; i<=72; i++)); do
    j=$(printf "%03d" $i)
    gsutil -m cp -r gs://i-care-2.0.physionet.org/training/*/*_${j}_EEG.* .
done
