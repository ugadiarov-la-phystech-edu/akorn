def load_data(data, data_root, episode_folder_pattern, data_imsize, is_eval=False, kind=None, image_file_extension=None, sequence_length=1):
    # Data loading
    
    collate_fn = None
    split = 'test' if is_eval else 'train'
    if kind is None:
        kind = 'image' if is_eval else 'pair'
    elif kind not in ('image', 'pair', 'video'):
        raise ValueError(f'Invalid dataset kind: {kind}')
    
    if data == "clevrtex_full" or data == "clevrtex_camo" or data == "clevrtex_outd":
        from source.data.datasets.objs.clevr_tex import get_clevrtex_pair, get_clevrtex, collate_fn
        if data == "clevrtex_camo":
            assert is_eval, "Camo dataset is only for evaluation"
            data_type = "camo"
            default_path = "./data/clevr_tex/clevrtex_camo"
        elif data == "clevrtex_outd":
            assert is_eval, "OOD dataset is only for evaluation"
            data_type = "outd"
            default_path = "./data/clevr_tex/clevrtex_outd"
        else:
            data_type = "full"
            default_path = "./data/clevr_tex/clevrtex_full"
            
        data_root = (
            default_path
            if data_root is None
            else data_root
        )
        imsize = 128 if data_imsize is None else data_imsize

        if kind == 'image':
            dataset = get_clevrtex(data_root, split=split, data_type=data_type, imsize=imsize, return_meta_data=True)
        elif kind == 'pair':
            dataset = get_clevrtex_pair(root=data_root, split=split, )
        else:
            raise ValueError(f'Unsupported dataset kind: {kind}')
            
    elif data == "clevr":
        from source.data.datasets.objs.clevr import get_clevr_pair, get_clevr
        imsize = 128 if data_imsize is None else data_imsize
        data_root = "./data/clevr_with_masks/"
        if kind == 'image':
            dataset = get_clevr(data_root, split=split, imsize=imsize)
        elif kind == 'pair':
            dataset = get_clevr_pair(root="./data/clevr_with_masks/", split=split, )
        else:
            raise ValueError(f'Unsupported dataset kind: {kind}')

    elif data == "imagenet":
        from source.data.datasets.objs.imagenet import get_imagenet_pair
        data_root = (
            "./data/ImageNet2012/"
            if data_root is None
            else data_root
        )
        dataset = get_imagenet_pair(
            root=data_root,
            split="train",
            hflip=True,
            imsize=256 if data_imsize is None else data_imsize,
        )
    elif data == "coco":
        imsize = 256 if data_imsize is None else data_imsize
        from source.data.datasets.objs.coco import get_coco_dataset
        data_root = "./data/COCO"
        if not is_eval:
            raise ValueError("COCO dataset is only for evaluation")
        else:
            dataset = get_coco_dataset(root=data_root)
        
    elif data == "pascal":
        imsize = 256 if data_imsize is None else data_imsize
        from source.data.datasets.objs.pascal import get_pascal_dataset
        data_root = "./data/VOCdevkit/VOC2012"
        if not is_eval:
            raise ValueError("Pascal dataset is only for evaluation")
        else:
            dataset = get_coco_dataset(root=data_root)
        
    elif data == "dsprites":
        imsize = 64 if data_imsize is None else data_imsize
        from source.data.datasets.objs.dsprites import get_dsprites_pair
        dataset = get_dsprites_pair(
            root="./data/multi_dsprites/",
            split="train",
            imsize=imsize,
        )
        
    elif data == "tetrominoes":
        from source.data.datasets.objs.tetrominoes import get_tetrominoes_pair
        imsize = 32 if data_imsize is None else data_imsize
        dataset = get_tetrominoes_pair(
            root="./data/tetrominoes/",
            split="train",
            imsize=imsize,
        )
        
    elif data == "Shapes":
        imsize = 40 if data_imsize is None else data_imsize
        from source.data.datasets.objs.shapes import get_shapes_pair
        dataset = get_shapes_pair(
            root="./data/Shapes/",
            split="train",
            imsize=imsize,
        )
    elif data == 'episode_dataset':
        from source.data.datasets.objs.episodes_dataset import EpisodesDataset, AugmentedPairEpisodeDataset
        kwargs = dict(root=data_root, episode_folder_pattern=episode_folder_pattern, mode=split, res=data_imsize,
                      extension=image_file_extension, kind=kind, sequence_length=sequence_length)
        if kind in ('image', 'video'):
            dataset = EpisodesDataset(**kwargs)
        elif kind == 'pair':
            for key in ('kind', 'sequence_length'):
                del kwargs[key]

            dataset = AugmentedPairEpisodeDataset(**kwargs)

        imsize = data_imsize
    elif data == 'image_dataset':
        from source.data.datasets.objs.image_dataset import ImageDataset, AugmentedPairImageDataset
        kwargs = dict(root=data_root, mode=split, res=data_imsize, extension=image_file_extension)
        if kind == 'image':
            dataset = ImageDataset(**kwargs)
        elif kind == 'pair':
            dataset = AugmentedPairImageDataset(**kwargs)

        imsize = data_imsize

    return dataset, imsize, collate_fn

