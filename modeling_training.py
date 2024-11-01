# @software{yolo11_ultralytics,
#   author = {Glenn Jocher and Jing Qiu},
#   title = {Ultralytics YOLO11},
#   version = {11.0.0},
#   year = {2024},
#   url = {https://github.com/ultralytics/ultralytics},
#   orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
#   license = {AGPL-3.0}
# }

if __name__ == '__main__':
    import os

    class Args:
        project = 'testing_code'
        config_train = 'config.yaml'
        epochs = 15
        batch = 4

    args = Args()


    from ultralytics import YOLO

    yolo_project = f'{args.project}/yolo11n'

    model = YOLO("yolo11n.yaml")
    model.to('cuda')
    # Train
    model.train(
        data=args.config_train, 
        epochs=args.epochs, 
        imgsz=640, 
        batch=args.batch, 
        project=yolo_project,
        freeze=10,
        patience=25,
        plots=True,
        half=False,
    )


    # Código de validação

    # Load a model
    model = YOLO( 
        os.path.join(yolo_project, 'train/weights/best.pt')
    )

    print(f'\n\nTEST ON {args.config_train}\n\n')

    # Validate the model
    metrics = model.val(
        data=args.config_train, 
        imgsz=640, 
        batch=1, 
        iou=0.5, 
        max_det=None,
        augment=False,
        # half=False,
        # dnn=False,
        device='cuda:0', 
        # rect=False,
        split='test',
        project=yolo_project
    )