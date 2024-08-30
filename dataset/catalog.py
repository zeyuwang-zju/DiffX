import os 

class DatasetCatalog:
    def __init__(self, ROOT):


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 


        self.VGGrounding = {   
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params": dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/gqa/tsv/train-00.tsv'),
            ),
        }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 


        self.FlickrGrounding = {
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/flickr30k/tsv/train-00.tsv'),
            ),
        }


        self.FlirGrounding = {
            "target": "dataset.flir_dataset.FLIRDataset",
            "train_params":dict(
                flir_path=ROOT,
                train_or_test = 'train',
            ),
            "val_params":dict(
                flir_path=ROOT,
                train_or_test = 'test',
            ),
        }


        self.MFNetGrounding = {
            "target": "dataset.mfnet_dataset.MFNetDataset",
            "train_params":dict(
                mfnet_path=ROOT,
                train_or_test = 'train',
            ),
            "val_params":dict(
                mfnet_path=ROOT,
                train_or_test = 'test',
            ),
        }


        self.MFNetTripleGrounding = {
            "target": "dataset.mfnet_triple_dataset.MFNetTripleDataset",
            "train_params":dict(
                mfnet_path=ROOT,
                train_or_test = 'train',
            ),
            "val_params":dict(
                mfnet_path=ROOT,
                train_or_test = 'test',
            ),
        }


        self.SODGrounding = {
            "target": "dataset.sod_dataset.SODDataset",
            "train_params":dict(
                sod_path=ROOT,
                train_or_test = 'train',
            ),
            "val_params":dict(
                sod_path=ROOT,
                train_or_test = 'test',
            ),
        }

        self.SobelGrounding = {
            "target": "dataset.sobel_dataset.SobelDataset",
            "train_params":dict(
                sod_path=ROOT,
                train_or_test = 'train',
            ),
            "val_params":dict(
                sod_path=ROOT,
                train_or_test = 'test',
            ),
        }

        self.SODSobelGrounding = {
            "target": "dataset.sod_sobel_dataset.SODSobelDataset",
            "train_params":dict(
                sod_path=ROOT,
                train_or_test = 'train',
            ),
            "val_params":dict(
                sod_path=ROOT,
                train_or_test = 'test',
            ),
        }


        self.MCXFaceGrounding = {
            "target": "dataset.mcxface_dataset.MCXFaceDataset",
            "train_params":dict(
                mcxface_path=ROOT,
                train_or_test = 'train',
            ),
            "val_params":dict(
                mcxface_path=ROOT,
                train_or_test = 'test',
            ),
        }

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 

        self.SBUGrounding = {   
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/SBU/tsv/train-00.tsv'),
            ),
         }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 


        self.CC3MGrounding = {   
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/CC3M/tsv/train-00.tsv'),
            ),
        }



        self.CC3MGroundingHed = {
            "target": "dataset.dataset_hed.HedDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/CC3M/tsv/train-00.tsv'),
                hed_tsv_path=os.path.join(ROOT,'GROUNDING/CC3M/tsv_hed/train-00.tsv'),
            ),
        }


        self.CC3MGroundingCanny = {
            "target": "dataset.dataset_canny.CannyDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/CC3M/tsv/train-00.tsv'),
                canny_tsv_path=os.path.join(ROOT,'GROUNDING/CC3M/tsv_canny/train-00.tsv'),
            ),
        }


        self.CC3MGroundingDepth = {
            "target": "dataset.dataset_depth.DepthDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/CC3M/tsv/train-00.tsv'),
                depth_tsv_path=os.path.join(ROOT,'GROUNDING/CC3M/tsv_depth/train-00.tsv'),
            ),
        }



        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 


        self.CC12MGrounding = {   
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/CC12M/tsv/train-00.tsv'),
            ),
        }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 

        self.Obj365Detection = {   
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'OBJECTS365/tsv/train-00.tsv'),
            ),
        }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 

        self.COCO2017Keypoint = {   
            "target": "dataset.dataset_kp.KeypointDataset",
            "train_params":dict(
                image_root = os.path.join(ROOT,'COCO/images'),
                keypoints_json_path = os.path.join(ROOT,'COCO/annotations2017/person_keypoints_train2017.json'),
                caption_json_path = os.path.join(ROOT,'COCO/annotations2017/captions_train2017.json'),
            ),
        }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 

        self.DIODENormal = {   
            "target": "dataset.dataset_normal.NormalDataset",
            "train_params":dict(
                image_rootdir = os.path.join(ROOT,'normal/image_train'),
                normal_rootdir = os.path.join(ROOT,'normal/normal_train'),
                caption_path = os.path.join(ROOT,'normal/diode_cation.json'),
            ),
        }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 

        self.ADESemantic = {   
            "target": "dataset.dataset_sem.SemanticDataset",
            "train_params":dict(
                image_rootdir = os.path.join(ROOT,'ADE/ADEChallengeData2016/images/training'),
                sem_rootdir = os.path.join(ROOT,'ADE/ADEChallengeData2016/annotations/training'),
                caption_path = os.path.join(ROOT,'ADE/ade_train_images_cation.json'),
            ),
        }





