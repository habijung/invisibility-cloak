# Invisibility Cloak OpenCV with Deep Learning

Invisibility Cloak Project

Computer Vision with OpenCV

Graduation Project in 2021 Fall Semester @UNIST

For this project, FGVC(Flow-edge Guided Video Completion) deep learning model was used for object removal.


# Result

### ✔ [Youtube](https://youtu.be/G4LIoRuKcas) | [Report (Kor)](/result/Invisibility_Cloak_Report.md)

<div align="center">
    <img src="https://filedn.com/ldHU78JSYWjSTua64JhwbGm/GitHub/invisibility-cloak/f250_compare_min.gif">
</div>


# Environment

| Package        | Version    |
| :------        | :------    |
| anaconda (x64) | 4.10.3     |
| cuda           | 10.2.89    |
| matplotlib     | 3.4.3      |
| numpy          | 1.21.4     |
| opencv         | 4.5.4      |
| os             | Windows 10 |
| pip            | 21.0.1     |
| python         | 3.8.12     |
| pytorch        | 1.6.0      |
| scipy          | 1.6.2      |


# Usage

- Download and unzip [weight.zip](https://filedn.com/ldHU78JSYWjSTua64JhwbGm/GitHub/invisibility-cloak/weight.zip) into the `modules`.
- Prepare video sequences dataset of color and mask for project.
  (Data Samples : [tennis](https://filedn.com/ldHU78JSYWjSTua64JhwbGm/GitHub/invisibility-cloak/data_tennis.zip) | [f250](https://filedn.com/ldHU78JSYWjSTua64JhwbGm/GitHub/invisibility-cloak/data_f250.zip))
- Run project

```sh
# Remove __pycache__.
$ find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

# Run video inpainting.
$ python run_inpainting.py \
> --path './data/tennis_color' \
> --path_mask './data/tennis_mask' \
> --outroot './data/tennis_result' \
> --merge \
> --run
```


# Update History

### v1.3

- Add to remove object with [FGVC](https://github.com/vt-vl-lab/FGVC)
- Set the main cloak color with **RED**
- Update cloak mask and noise.
- Rename **ftn -> image_stack**.

### v1.2

- Add to save output video.
- Add color selection mode : **RED || GREEN**
- Get object removal result by [FGVC](https://github.com/vt-vl-lab/FGVC).

### v1.1

- Add specific noise filtering conditions.
- Modify detected color : **RED -> GREEN**

### v1.0

- Project first commit.
- Test module **ftn** for showing image stack view.
- Test **HSV Detector** using track bar.
- Test **Invisibility Cloak** Demo.


<br>

---
**Updated :** 2021-11-28 00:19
