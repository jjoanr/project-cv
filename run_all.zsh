# shell script to automate the evaluation with the collected dataset
for i in {1..12}; do
    python inference.py \
        --image "Dataset/Figures_reis/Avaluacio_reis/${i}.jpg" \
        --weights "model_output/model_100img_layer4.pth" \
        --backbone resnet18 \
        --support-dir "Dataset/Imatges_propies/peces_reis" \
        --k-shot 10 \
        --crop-scale 1.4
done
