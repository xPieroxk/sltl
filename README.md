# SLTL - Master's Thesis

This repository contains the code and experiments conducted as part of my Master's thesis.

## Setup

1. **Clone the repository and navigate to it**:  
   `git clone https://github.com/xPieroxk/sltl.git`

   `cd sltl`

2. **Unzip the dataset**:  
   Before running the evaluation script, make sure to unzip the `datasets` folder.

3. **Create and activate a virtual environment**:  
   `python -m venv venv && source venv/bin/activate`  
   *(On Windows: `venv\Scripts\activate`)*

4. **Install the required packages**:  
   `pip install -r requirements.txt`

## Running the Script

To generate plots, run `plot.py` with one or two arguments:  
`python plot.py <i> [<j>]`

- `<i>` and `<j>` are patient indices ranging from **1 to 210** (inclusive).


### Examples


- `python plot.py 5` — generates the plot for **patient 5**
- `python plot.py 3 17` — compares **patient 3** with **patient 17**