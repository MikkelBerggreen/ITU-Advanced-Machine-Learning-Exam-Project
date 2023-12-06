from imports import *

def download_data(fnames, urls):
    for fname, url in zip(fnames, urls):
        if not os.path.isfile(fname):
            try:
                r = requests.get(url)
            except requests.ConnectionError:
                print("!!! Failed to download data !!!")
                return False
            else:
                if r.status_code != requests.codes.ok:
                    print("!!! Failed to download data !!!")
                    return False
                else:
                    print(f"Downloading {fname}...")
                    with open(fname, "wb") as fid:
                        fid.write(r.content)
                    print(f"Download {fname} completed!")
    return True

def load_data(fname):
    with np.load(fname) as dobj:
        dat = dict(**dobj)
    labels = np.load('../kay_labels.npy')
    val_labels = np.load('../kay_labels_val.npy')

    training_inputs = dat['stimuli']
    test_inputs = dat['stimuli_test']
    training_outputs = dat['responses']
    test_outputs = dat['responses_test']
    roi = dat['roi']
    roi_names = dat['roi_names']

    return training_inputs, test_inputs, training_outputs, test_outputs, roi, roi_names, labels, val_labels
