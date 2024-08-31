import os
import argparse

current_file_path = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Creates the folders")
parser.add_argument("folder_path", type=str, 
                    help="Enter the folder path where you want to create the folders")

args = parser.parse_args()

project_dir_path = args.folder_path

def create_folders(project_dir_path):
    if os.path.exists(project_dir_path):
        pass
    else:
        os.mkdir(project_dir_path)

    for folder in ["Code","Data"]:
        if os.path.exists(os.path.join(project_dir_path,folder)):
            print(f"{folder} folder already exists")
            pass
        else:
            os.mkdir(os.path.join(project_dir_path,folder))
            print(f"{folder} folder created succesfully")
        
    for folder_ in ["Raw","Processed","Results"]:
        if os.path.exists(os.path.join(project_dir_path,"Data",folder_)):
            print(f"{folder_} folder already exists")
            pass
        else:
            os.mkdir(os.path.join(project_dir_path,"Data",folder_))
            print(f"{folder_} folder created succesfully")
    
    for folder__ in ["Notebooks","Utils"]:
        if os.path.exists(os.path.join(project_dir_path,"Code",folder__)):
            print(f"{folder__} folder already exists")
            pass
        else:
            os.mkdir(os.path.join(project_dir_path,"Code",folder__))
            print(f"{folder__} folder created succesfully")
    
    print("All folders created")


def create_constants_file(project_dir_path):
    content = """import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR,"Data")
RAW_DIR = os.path.join(DATA_DIR,"Raw")
PROCESSED_DIR = os.path.join(DATA_DIR,"Processed")
RESULTS_DIR = os.path.join(DATA_DIR,"Results")
              """
    with open(os.path.join(project_dir_path,"Code","Utils","constants.py"), "w") as f:
        f.write(content)
    f.close()

    print("Created constants.py file successfully.")

if __name__ == "__main__":
    create_folders(project_dir_path)
    create_constants_file(project_dir_path)