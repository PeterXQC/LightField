for ($i=0; $i -le 9; $i++)
{
    $image_file = "data/" + $i + ".png"
    python test.py --weights-file "rdn_x4.pth" --image-file $image_file --scale 4 --num-features 64 --growth-rate 64 --num-blocks 16 --num-layers 8
}

for ($i=0; $i -le 9; $i++)
{
    $image_file = "data/" + $i + ".png"
    python test.py --weights-file "rdn_x2.pth" --image-file $image_file --scale 2 --num-features 64 --growth-rate 64 --num-blocks 16 --num-layers 8
}