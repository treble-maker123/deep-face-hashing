# Face Hasing Using Neural Networks

## File structure

```bash
|--- code/ # contains code for the project
|    |--- data/ # contains preprocessed data
|--- facescrub/ # from https://github.com/faceteam/facescrub.git
|    |--- download/ # data from the download.py script
|    |--- download.py # script to download data
```

## Packages
Run `source activate cs670project` to activate project conda environment.

### Package Versions
- python=3.6
- numpy=1.15.4
- scipy=1.1.0
- opencv-python=3.4.3.18
- matplotlib=3.0.1
- jupyter=1.0.0

### Installing Pytorch
`conda install pytorch torchvision -c pytorch`

## Miscellaneous Notes

- Location of dataset https://github.com/faceteam/facescrub.git. **NOTE:** Need Python 2.7 to run download.py.

## Instructions

1. Download miniconda and pip,
2. Install the packages noted above,
3. `git clone https://github.com/faceteam/facescrub.git` into the same level as the project's path (see file structure above),
4. Run `python download.py` with Python 2.7 to download the FaceScrub images,
5. Run `python utils.py` to preprocess the images and move them into the project's ./data folder,
