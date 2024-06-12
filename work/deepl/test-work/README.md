## 用于获取两套数据集的编号和分类

hmdb51存在51种动作分类，ucf101存在101种动作分类

数据集文件在list目录下

`get_classify.py` 可用于获取数据集中的每个编号对应的动作名称，以及该类动作的数据个数

运行效果：

```bash
$python get_classify.py 
hmdb51:
[70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70]
['smile', 'clap', 'climb', 'cartwheel', 'pushup', 'push', 'somersault', 'turn', 'walk', 'shake_hands', 'pick', 'chew', 'jump', 'pour', 'smoke', 'shoot_bow', 'swing_baseball', 'kick', 'catch', 'golf', 'dribble', 'draw_sword', 'laugh', 'ride_horse', 'fall_floor', 'stand', 'sword', 'shoot_gun', 'kiss', 'eat', 'sword_exercise', 'flic_flac', 'handstand', 'brush_hair', 'pullup', 'throw', 'sit', 'shoot_ball', 'fencing', 'run', 'wave', 'drink', 'situp', 'punch', 'hit', 'ride_bike', 'kick_ball', 'hug', 'climb_stairs', 'dive', 'talk']
ucf101:
[101, 82, 104, 97, 77, 112, 107, 99, 94, 112, 96, 110, 93, 76, 82, 112, 114, 97, 73, 95, 79, 99, 103, 118, 77, 105, 116, 77, 86, 89, 89, 100, 100, 97, 107, 105, 100, 77, 106, 86, 89, 115, 91, 112, 86, 81, 86, 106, 105, 89, 92, 90, 92, 91, 76, 97, 77, 80, 120, 110, 115, 107, 117, 77, 113, 80, 72, 109, 88, 72, 121, 72, 83, 103, 85, 101, 90, 118, 98, 88, 95, 72, 79, 108, 96, 80, 82, 93, 89, 101, 72, 117, 92, 87, 93, 76, 81, 87, 95, 107, 92]
['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']
```
