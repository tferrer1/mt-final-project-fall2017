if [ $1 == 1 ] ; then
    echo "running train.py"
    python train.py --batch_size 50 --epochs 40 --gpuid 1
    # python train.py --data_file data/hw4_data.bin --optimizer Adam -lr 1e-2 --batch_size 48 --model_file model.py
elif [ $1 == 2 ] ; then
    echo "running train_bi.py"
    python train_bi.py --data_file data/hw4_data.bin --optimizer Adam -lr 1e-2 --batch_size 48 --model_file model.py
else
    echo "Unknown parameter"
fi

