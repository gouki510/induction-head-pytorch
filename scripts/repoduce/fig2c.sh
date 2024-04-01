# python train.py --ways 1 --num_class 1024 --eps 0 --device cuda:1 --p_bursty 1 --p_icl 0 --exp_name fig2b
python train.py --ways 1 --num_class 1024 --eps 0.1 --device cuda:2 --p_bursty 0.5 --p_icl 0 --exp_name fig2c --alpha 0
python train.py --ways 1 --num_class 1024 --eps 0.5 --device cuda:2 --p_bursty 0.5 --p_icl 0 --exp_name fig2c --alpha 0.5
python train.py --ways 1 --num_class 1024 --eps 0.75 --device cuda:2 --p_bursty 0.5 --p_icl 0 --exp_name fig2c --alpha 1
python train.py --ways 1 --num_class 1024 --eps 0 --device cuda:2 --p_bursty 0.5 --p_icl 0 --exp_name fig2c --alpha 1.5
