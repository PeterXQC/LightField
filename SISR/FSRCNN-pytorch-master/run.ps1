for ($i=0; $i -le 9; $i++)
{
    $image_file = "data/" + $i + ".png"
    python test.py --weights-file "fsrcnn_x4.pth" --image-file $image_file --scale 4
}

for ($i=0; $i -le 9; $i++)
{
    $image_file = "data/" + $i + ".png"
    python test.py --weights-file "fsrcnn_x2.pth" --image-file $image_file --scale 2
}