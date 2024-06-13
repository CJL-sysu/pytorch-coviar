import argparse
import pickle
import numpy as np
import torch
from transforms import GroupCenterCrop
from transforms import GroupOverSample
from transforms import GroupScale
from model import Model
import torchvision

def load_list_from_bin_file(file_path):
    with open(file_path, 'rb') as f:
        list_ = pickle.load(f)
    return list_

def get_input(file_path, transform, representation):
    """
        representation必须和client一致
    """
    frames = load_list_from_bin_file(file_path)
    frames = transform(frames)
    frames = np.array(frames)
    frames = np.transpose(frames, (0, 3, 1, 2))
    input = torch.from_numpy(frames).float() / 255.0
    
    input_mean = torch.from_numpy(
        np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
    input_std = torch.from_numpy(
        np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()
    
    if representation == 'iframe':
        input = (input - input_mean) / input_std
    elif representation == 'residual':
        input = (input - 0.5) / input_std
    elif representation == 'mv':
        input = input - 0.5
    return input

def parse_args():
    # parse args
    # test_segments和representation必须和client端保持一致
    parser = argparse.ArgumentParser(description="classify video")
    parser.add_argument("--data_name", type=str, choices=["ucf101", "hmdb51"])
    parser.add_argument("--representation", type=str, choices=["iframe", "residual", "mv"])
    parser.add_argument("--test_segments", type=int, default=25)
    parser.add_argument("--arch", type=str)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--test-crops", type=int, default=10)
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--gpus", nargs="+", type=int, default=None)
    parser.add_argument(
        "-j",
        "--workers",
        default=1,
        type=int,
        metavar="N",
        help="number of workers.",
    )

    args = parser.parse_args()
    return args

def main(args):
    if args.data_name == 'ucf101':
        num_class = 101
        classes = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']
    elif args.data_name == 'hmdb51':
        num_class = 51
        classes = ['smile', 'clap', 'climb', 'cartwheel', 'pushup', 'push', 'somersault', 'turn', 'walk', 'shake_hands', 'pick', 'chew', 'jump', 'pour', 'smoke', 'shoot_bow', 'swing_baseball', 'kick', 'catch', 'golf', 'dribble', 'draw_sword', 'laugh', 'ride_horse', 'fall_floor', 'stand', 'sword', 'shoot_gun', 'kiss', 'eat', 'sword_exercise', 'flic_flac', 'handstand', 'brush_hair', 'pullup', 'throw', 'sit', 'shoot_ball', 'fencing', 'run', 'wave', 'drink', 'situp', 'punch', 'hit', 'ride_bike', 'kick_ball', 'hug', 'climb_stairs', 'dive', 'talk']
    else:
        raise ValueError('Unknown dataset '+args.data_name)
    net = Model(
        num_class, args.test_segments, args.representation, base_model=args.arch
    )  # 使用预训练模型resnet构建网络
    checkpoint = torch.load(args.weights)  # 加载训练好的模型参数
    # print(
    #     "model epoch {} best prec@1: {}".format(
    #         checkpoint["epoch"], checkpoint["best_prec1"]
    #     )
    # )
    base_dict = {
        ".".join(k.split(".")[1:]): v for k, v in list(checkpoint["state_dict"].items())
    }
    net.load_state_dict(base_dict)  # 导入模型参数
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose(
            [
                GroupScale(net.scale_size),
                GroupCenterCrop(net.crop_size),
            ]
        )
    elif args.test_crops == 10:  # 默认是10
        cropping = torchvision.transforms.Compose(
            [
                GroupOverSample(
                    net.crop_size, net.scale_size, is_mv=(args.representation == "mv")
                )
            ]
        )
    else:
        raise ValueError(
            "Only 1 and 10 crops are supported, but got {}.".format(args.test_crops)
        )
    if args.file_path is None:
        raise ValueError("You must specify a file path")
    inp = get_input(file_path=args.file_path, transform=cropping, representation=args.representation)
    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))
    net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
    net.eval()
    def forward_video(data):
        input_var = torch.autograd.Variable(data, volatile=True)
        scores = net(input_var)
        scores = scores.view(
            (-1, args.test_segments * args.test_crops) + scores.size()[1:]
        )
        scores = torch.mean(scores, dim=1)
        return scores.data.cpu().numpy().copy()
    video_scores = forward_video(inp)
    classify_num = np.argmax(video_scores)
    return (classify_num, classes[classify_num])


if __name__ == "__main__":
    args = parse_args()
    classify_num, classify = main(args)
    print("the classify result is {}#{}".format(classify_num, classify))
