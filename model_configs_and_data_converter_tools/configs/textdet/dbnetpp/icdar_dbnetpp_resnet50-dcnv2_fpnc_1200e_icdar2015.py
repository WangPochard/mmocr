_base_ = [
    '_base_dbnetpp_resnet50-dcnv2_fpnc.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/datasets/ic100-2.py',
    '../_base_/datasets/ctw1500.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textdet/dbnetpp/tmp_1.0_pretrain/dbnetpp_r50dcnv2_fpnc_100k_iter_synthtext-20220502-352fec8a.pth'  # noqa
#'C:/Users/User/Desktop/C_VSCode/mmocr/mmocr/tools/data/ic100-2/pretrain380.pth'  # noqa

# dataset settings
train_list = [_base_.icdar2015_textdet_train,
              _base_.ic100_textdet_train,
              _base_.ctw1500_textdet_train]
test_list = [_base_.icdar2015_textdet_test,
             _base_.ic100_textdet_test]
            #  _base_.ctw1500_textdet_test]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=4)
