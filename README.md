# Gaussian Mixture GAN - DataLabAssignement2

## IASD Master Program 2023/2024 - PSL Research University

### About this project

This project is the second homework assignment for the Data Science Lab class of the IASD (Artificial Intelligence, Systems and Data) Master Program 2023/2024, at PSL Research University (Université PSL).

*The project achieved the following objectives:*
- Improved the digit generation of the MNIST dataset by employing Gaussian Mixture Generative Adversarial Networks (GM-GAN) in unsupervised learning scenarios, expanding on a well-optimized vanilla GAN.
- Established a strong baseline with the vanilla GAN through meticulous hyperparameter adjustments, providing valuable insights for refining a static GM-GAN.
- Generated a more diverse set of aesthetically pleasing digits at a higher frequency using the final Gaussian Mixture model and recommended hyperparameters identified through the tuning analysis.

## General information

The report can be viewed in the [report.pdf](report.pdf) file. It answers to the instructions given in the [assignment_2_slides_instructions.pdf](assignment_2_slides_instructions.pdf) file provided by the professors.

The rest of the instructions can be found below. If you want to copy and recreate this project, or test it for yourself, some important information to know.

**generate.py**
Use the file *generate.py* to generate 10000 samples of MNIST in the folder samples. 
Example:
  > python3 generate.py --bacth_size 64

**requirements.txt**
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different librairies you might use. 
When your code will be test, we will execute: 
  > pip install -r requirements.txt

**Checkpoints**
Push the minimal amount of models in the folder *checkpoints*.

---

### Acknowledgement

This project was made possible with the guidance and support of the following :

- **Prof. Benjamin Negrevergne**
  - Professor at *Université Paris-Dauphine, PSL*
  - Researcher in the *MILES Team* at *LAMSADE, UMR 7243* at *Université Paris-Dauphine, PSL* and *Université PSL*
  - Co-director of the IASD Master Program with Olivier Cappé

- **Alexandre Vérine**
  - PhD candidate at *LAMSADE, UMR 7243* at *Université Paris-Dauphine, PSL* and *Université PSL*
 
This project was a group project, and was made possible thanks to the collaboration of :

- **Jules Merigot**, *IASD Master Program 2023/2024 student, at PSL Research University*
- **Mattéo Sammut**, *IASD Master Program 2023/2024 student, at PSL Research University*

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

**Note:**

This project is part of ongoing research and is subject to certain restrictions. Please refer to the license section and the [LICENSE.md](LICENSE.md) file for details on how to use this code for research purposes.
