
# For the case: labeled = 300

# Input mixup
python latent-mixing.py --augu --out 'Final_models/ip1@300' --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 300 --mixup 'input' --alpha 1.0 --manualSeed 1 --noSharp --gpu 0
python latent-mixing.py --augu --out 'Final_models/ip2@300' --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 300 --mixup 'input' --alpha 1.0 --manualSeed 2 --noSharp --gpu 0
python latent-mixing.py --augu --out 'Final_models/ip3@300' --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 300 --mixup 'input' --alpha 1.0 --manualSeed 3 --noSharp --gpu 0
python latent-mixing.py --augu --out 'Final_models/ip4@300' --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 300 --mixup 'input' --alpha 1.0 --manualSeed 4 --noSharp --gpu 0
python latent-mixing.py --augu --out 'Final_models/ip5@300' --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 300 --mixup 'input' --alpha 1.0 --manualSeed 5 --noSharp --gpu 0

# Input + Latent mixup
python latent-mixing.py --augu --out 'Final_models/mn1@300' --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 300 --mixup 'mixup_hidden' --alpha 2.0 --manualSeed 1 --noSharp --gpu 0
python latent-mixing.py --augu --out 'Final_models/mn2@300' --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 300 --mixup 'mixup_hidden' --alpha 2.0 --manualSeed 2 --noSharp --gpu 0
python latent-mixing.py --augu --out 'Final_models/mn3@300' --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 300 --mixup 'mixup_hidden' --alpha 2.0 --manualSeed 3 --noSharp --gpu 0
python latent-mixing.py --augu --out 'Final_models/mn4@300' --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 300 --mixup 'mixup_hidden' --alpha 2.0 --manualSeed 4 --noSharp --gpu 0
python latent-mixing.py --augu --out 'Final_models/mn5@300' --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 300 --mixup 'mixup_hidden' --alpha 2.0 --manualSeed 5 --noSharp --gpu 0

# Only Latent mixup
python latent-mixing.py --augu --out 'Final_models/oh1@300' --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 300 --mixup 'only_hidden' --alpha 2.0 --manualSeed 1 --noSharp --gpu 0
python latent-mixing.py --augu --out 'Final_models/oh2@300' --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 300 --mixup 'only_hidden' --alpha 2.0 --manualSeed 2 --noSharp --gpu 0
python latent-mixing.py --augu --out 'Final_models/oh3@300' --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 300 --mixup 'only_hidden' --alpha 2.0 --manualSeed 3 --noSharp --gpu 0
python latent-mixing.py --augu --out 'Final_models/oh4@300' --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 300 --mixup 'only_hidden' --alpha 2.0 --manualSeed 4 --noSharp --gpu 0
python latent-mixing.py --augu --out 'Final_models/oh5@300' --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 300 --mixup 'only_hidden' --alpha 2.0 --manualSeed 5 --noSharp --gpu 0



