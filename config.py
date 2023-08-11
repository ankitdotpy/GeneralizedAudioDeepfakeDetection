import glob

LA_TRAIN_PROTOCOL = '/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
LA_DEV_PROTOCOL = '/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
LA_EVAL_PROTOCOL = '/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'

LA_TRAIN_DIR = '/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_train/flac/'
LA_DEV_DIR = '/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_dev/flac/'
LA_EVAL_DIR = '/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_eval/flac/'

NOISE_DIR = '/kaggle/input/birdclef2021-background-noise/ff1010bird_nocall/nocall'

class config:
    sr = 16000
    duration = 4 # seconds
    samples = sr * duration
    n_fft = 1024
    window = 512
    hop_length = 256
    n_mels = 128
    fmin = 20
    fmax = sr//2
    top_db = 80
    freq_param = 10
    noise_files = glob.glob(NOISE_DIR+'/*.ogg')