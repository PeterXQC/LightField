$n = Read-Host "Please enter the number n"
$s = Read-Host "Please enter the scale s"

for ($i=0; $i -le $n; $i++){
    $inputFile = "..\LF\$i.jpg"
    $outputFile = ".\output\$s\$i.jpg"
    & ".\realesrgan-ncnn-vulkan.exe" -i $inputFile -o $outputFile -s $s
}