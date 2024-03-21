# mmocr
Fine Tuning [MMOCR](https://github.com/open-mmlab/mmocr)
mmocr-rec使用AbiNet Union14m 在[MMOCR](https://github.com/open-mmlab/mmocr) 中能找到該模型檔
---

### Linux 環境建置
1. Pytorch
    
    ```python
    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
    ```
2. glx
    
    ```python
    sudo apt-get install libgl1-mesa-glx
    ```
    
3. cudnn 安裝  [source](https://discuss.pytorch.org/t/libcudnn-cnn-infer-so-8-library-can-not-found/164661/26)
    
    ```python
    pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
    
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/python3.10/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib
    ```
    
4. mmocr 安裝
    
    ```python
    pip install -U openmim
    mim install mmengine
    pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
    pip install mmdet==3.0.0rc5
    
    ```
    
    ```python
    git clone https://github.com/open-mmlab/mmocr.git
    cd mmocr
    git checkout dev-1.x
    pip install -v -e .
    
    >>> from mmocr.apis import MMOCRInferencer
    >>> ocr = MMOCRInferencer(det='DBNet', rec='CRNN')
    >>> ocr('demo/demo_text_ocr.jpg', show=False, print_result=True)
    ```
    
5. Google Drive
    
    ```python
    pip install gdown
    sudo apt-get install unzip
    ```
    
6. mmocr  資料訓練
    1. data prepreation icdar2015
        
        ```python
        (dir to detext_converter.py path)
        python detext_converter.py path/to/.../icdar/ --nproc 4
        
        ```
        ~~python tools/dataset_converters/prepare_dataset.py icdar2015 --task textdet~~
        ~~python tools/dataset_converters/prepare_dataset.py totaltext --task textdet~~
        ~~python tools/dataset_converters/prepare_dataset.py ctw1500 --task textdet~~ 
    2. training
        
        ```python
        python tools/train.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py --work-dir dbnet/ --amp
        ```
        
        ```python
        
        tools/dist_train.sh configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py 8
        ```
