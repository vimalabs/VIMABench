# Datasheet
We present a [Datasheet](https://dl.acm.org/doi/10.1145/3458723) for documentation and responsible usage of our training dataset.

## Motivation
### For what purpose was the dataset created? 
We create this dataset to learn general robot manipulation with multimodal prompts.

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?
This dataset was created by Yunfan Jiang (NVIDIA, Stanford), Agrim Gupta (Stanford), Zichen "Charles" Zhang (Macalester College), Guanzhi Wang (NVIDIA, Caltech), Yongqiang Dou (Tsinghua), Yanjun Chen (Stanford), Li Fei-Fei (Stanford), Anima Anandkumar (NVIDIA, Caltech), Yuke Zhu (NVIDIA, UT Austin), and Linxi "Jim" Fan (NVIDIA).

## Distribution
### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?
Yes, the dataset is publicly available on the internet.

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?
The dataset can be downloaded from [🤗Hugging Face](https://huggingface.co/datasets/VIMA/VIMA-Data).

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?
No.

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?
No.

## Maintenance
### Who will be supporting/hosting/maintaining the dataset?
The authors will be supporting, hosting, and maintaining the dataset.

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?
Please contact Yunfan Jiang (yunfanjyf@gmail.com) and Linxi Fan (linxif@nvidia.com).

### Is there an erratum?
No. We will make announcements if there is any.

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?
Yes. New updates will be posted on https://vimalabs.github.io/.

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were the individuals in question told that their data would be retained for a fixed period of time and then deleted)?
N/A.

### Will older versions of the dataset continue to be supported/hosted/maintained? 
Yes, old versions will be permanently accessible on 🤗Hugging Face.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?
Yes, please refer to https://vimalabs.github.io/.

## Composition
### What do the instances that comprise the dataset represent?
Our data contain successful demonstrations to complete robotics tasks paired with multimodal prompts. Data modalities include RGB images, arrays (e.g., for actions), and structured data (e.g., for task meta info).

### How many instances are there in total (of each type, if appropriate)?
We provide 650K successful trajectories in total.

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?
We provide all instances in our 🤗Hugging Face data repositories.

### Is there a label or target associated with each instance?
Yes, we provide optimal action labels to train behavior cloning models.

### Is any information missing from individual instances?
No.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?
N/A.

### Are there recommended data splits (e.g., training, development/validation, testing)?
Yes, we use 600K for training and 50K for validation.

### Are there any errors, sources of noise, or redundancies in the dataset?
No.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g.,websites, tweets, other datasets)?
Yes, it is self-contained.

### Does the dataset contain data that might be considered confidential?
No.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?
No.

## Collection Process
### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)? 
All data collection, curation, and filtering are done by VIMA coauthors.

### Over what timeframe was the data collected?
The data was collected primarily during summer 2022.

## Uses
### Has the dataset been used for any tasks already?
Yes, we have used it to train our VIMA models for general robot manipulation.

### What (other) tasks could the dataset be used for?
This dataset also serves the purpose to learn generalist agents.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?
No.

### Are there tasks for which the dataset should not be used?
No.
