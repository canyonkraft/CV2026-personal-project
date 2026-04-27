# CV2026-personal-project
## Included Files
* pipeline.py, where all the code is housed
* requirements.txt, used to install all required libraries
* Note that the datasets themselves are not included, please follow the instructions below to download them.

## Core Requirements
There are a lot of packages that need to be installed to run the program. These can be found in requirements.txt. Run the following command to install all the packages:
```
pip install -r requirements.txt
```

## Dataset Downloads
There are three datasets that I used during this project. Note that all three don't need to be downloaded for the project to work, however at least one is needed for the model to run. If you are downloading all three datasets, you will need about 3GB of disk space. Here are the three datasets used:
* Stanford Dogs Dataset: Navigate to [this site](http://vision.stanford.edu/aditya86/ImageNetDogs/) and install the images.tar and annotation.tar files.
* Oxford-IIIT Pets Dataset: Navigate to [this site](https://www.robots.ox.ac.uk/~vgg/data/pets/) and install the images.tar.gz and annotations.tar.gz files.
* StanfordExtra Annotations: Navigate to [this github repo](https://github.com/benjiebob/StanfordExtra) and scroll down until you see the 'Download' section. Click on the google form link and fill out the google form. Upon completion, you will be emailed with a link to the download of the json file.

These three datasets must be in the same directory as the other project files, under ~/data. Extract the Stanford Dogs images under ~/data/stanford_dogs/images and the annotations under ~/data/stanford_dogs/annotation. You may also drop the StanfordExtra json file into ~/data/stanford_dogs.
If you install the Oxford-IIIT dataset, extract the images.tar.gz file under ~/data/oxford_pets/images and the annotations.tar.gz file under ~/data/oxford_pets/annotations. This will let the model be able to see the dataset images and annotations and use them during training.

## Program Execution
The script pipeline.py is divided into multiple phases:
* Preprocess phase, where the program locates the dataset files
* Caption phase, where the program assigns captions to each image in the datasets
* Training phase, where the model trains on all the captioned images
* Generation phase, where the model outputs generated images under ~/outputs
* Optional: Evaluation, compare, and charts phase to analyze the output with FID and CLIP score.

To run the program, you must first install all the required packages, the instructions to do so are above. You must also have an NVIDIA GPU which is available to run cuda processes. Then, run each of these commands:
```
python pipeline.py --stage preprocess
python pipeline.py --stage caption
python pipeline.py --stage train
python pipeline.py --stage generate --breed "Golden Retriever" --num_images 8

# optional
python pipeline.py --stage evaluate --breed "Golden Retriever"
python pipeline.py --stage compare \
    --breeds "Golden Retriever" "Siberian Husky" "Chihuahua" "Beagle" "Border Collie" \
    --num_per_breed 8
python pipeline.py --stage charts
```

## Available Breeds to Output
The model can output any of these breeds:
* Golden Retriever, Labrador Retriever, Chesapeake Bay Retriever, Curly-Coated Retriever, Flat-Coated Retriever, English Setter, Irish Setter, Gordon Setter, German Short-Haired Pointer, Vizsla, Brittany Spaniel, Cocker Spaniel, English Springer Spaniel, Welsh Springer Spaniel, Sussex Spaniel, Clumber, Irish Water Spaniel, Weimaraner

* Afghan Hound, Basset, Beagle, Bloodhound, Bluetick, Black-And-Tan Coonhound, Borzoi, English Foxhound, Walker Hound, Ibizan Hound, Irish Wolfhound, Italian Greyhound, Norwegian Elkhound, Otterhound, Redbone, Saluki, Scottish Deerhound, Whippet

* Boxer, Bull Mastiff, Doberman, Eskimo Dog, French Bulldog, Great Dane, Great Pyrenees, Greater Swiss Mountain Dog, Bernese Mountain Dog, Entlebucher, Appenzeller, Komondor, Kuvasz, Leonberg, Newfoundland, Rottweiler, Saint Bernard, Tibetan Mastiff, Standard Schnauzer, Giant Schnauzer, Miniature 

* Alaskan Malamute, Siberian Husky, Samoyed, Chow, Keeshond, Pomeranian

* Airedale, American Staffordshire Terrier, Staffordshire Bullterrier, Australian Terrier, Bedlington Terrier, Border Terrier, Cairn, Dandie Dinmont, Irish Terrier, Kerry Blue Terrier, Lakeland Terrier, Norfolk Terrier, Norwich Terrier, Scotch Terrier, Sealyham Terrier, Silky Terrier, Soft-Coated Wheaten Terrier, Tibetan Terrier, Wire-Haired Fox Terrier, West Highland White Terrier, Yorkshire Terrier

* Border Collie, Collie, Shetland Sheepdog, Old English Sheepdog, Briard, Bouvier Des Flandres, German Shepherd, Groenendael, Malinois, Rhodesian Ridgeback, Cardigan, Pembroke

* Affenpinscher, Blenheim Spaniel, Brabancon Griffon, Chihuahua, Japanese Spaniel, Maltese Dog, Mexican Hairless, Miniature Pinscher, Papillon, Pekinese, Pug, Shih-Tzu, Toy Poodle, Miniature Poodle, Standard Poodle, Toy Terrier

* Basenji, Boston Bull, Dhole, Dingo, Schipperke, Lhasa, African Hunting Dog, English Bulldog