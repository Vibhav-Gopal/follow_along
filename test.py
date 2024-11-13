import warnings
warnings.filterwarnings("ignore")
import torch
import config
import torch.optim as optim

from model import YOLOv3

from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
    non_max_suppression,
    plot_image
)

torch.backends.cudnn.benchmark = True
def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)


    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path = config.DATASET+"/train.csv", test_csv_path = config.DATASET + "/test.csv"
    )
    checkpoint_file="from_server/new_checkpoint.pth.tar"
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        *torch.tensor(config.S).unsqueeze(1).unsqueeze(2).repeat(1,3,2)

    ).to(config.DEVICE)
    print("Model loaded")
    model.eval()
    # check_class_accuracy(model, test_loader, threshold=.5)
    # pred_boxes, true_boxes = get_evaluation_bboxes(
    #     test_loader,
    #     model,
    #     iou_threshold=config.NMS_IOU_THRESH,
    #     anchors=config.ANCHORS,
    #     threshold=config.CONF_THRESHOLD,
    # )
    # mapval = mean_average_precision(
    #     pred_boxes,
    #     true_boxes,
    #     iou_threshold=config.MAP_IOU_THRESH,
    #     box_format="midpoint",
    #     num_classes=config.NUM_CLASSES,
    # )
    # print(f"MAP: {mapval.item()}")
    # print("Done")
    plot_couple_examples(model, test_loader, 0.6, 0.1, scaled_anchors)
    # model.train()

if __name__ == "__main__":
    main()