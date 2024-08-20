mkdir data
cd data

wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip

unzip train2014.zip
unzip val2014.zip

rm train2014.zip
rm val2014.zip

cd ../
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip

unzip v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
unzip v2_Questions_Val_mscoco.zip

rm v2_Annotations_Train_mscoco.zip
rm v2_Annotations_Val_mscoco.zip
rm v2_Questions_Train_mscoco.zip
rm v2_Questions_Val_mscoco.zip