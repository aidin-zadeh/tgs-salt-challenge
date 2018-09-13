# unet

## Background

## Methodology

## Data

## Report

## Requirements
- cython                        0.28.5
- numpy                         1.15.1 
- tensorflow-gpu                1.10.1
- jupyter                       1.0.0
- nb_conda                      2.2.1
- matplotlib                    2.2.3
- pillow                        5.2.0 

## Directory Structure
```
.
├── unet	<- source files used in this project
│   ├── conf		<- data utilized in this project
│   ├── data
│   │   ├── ext
│   │   ├── int
│   │   └── raw
│   └── scripts		<- Scripts used in this project.
├── docs		<- Documents related to this project
├── images		<- Images for README.md files
├── notebooks		<- Ipython notebook files
└── reports		<- Generated analysis as HTML, PDF, Latex, etc
    ├── figures		<- Generated graphics and figures used in reporting
    └── logs		<- Generated log files
```
## Installation
Install python dependencies from  `requirements.txt` using conda.
```bash
conda install --yes --file conda-requirements.txt
```

Or create a new conda environment `<new-env-name>` by importing a copy of a working conda environment `conda-tfunet` at the project root directory.
```bash
conda env create --name <new-env-name> -f conda-tfunet.yml
```
## Usage

## References

## To Do
- [ ] TBA

## License
MIT License

