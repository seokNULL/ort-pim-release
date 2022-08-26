cp ../setup.py .
cp -r  ../docs/ .
cp ../requirements.txt .
## ARM
#python3 setup.py bdist_wheel -p linux_aarch64
# x86
#python3 setup.py bdist_wheel -p linux_x86_64 --use_cuda
rm -r ./dist
python3 setup.py bdist_wheel -p linux_x86_64 && yes | pip uninstall onnxruntime && pip install dist/*.whl
