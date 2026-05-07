# INSTRUCTIONS

## Website with quarto

This repository is already structured as a quarto website with the following pages:

- `index.qmd`: Landing/front page of the website. It shall have a short **Executive Summary**. 
- `introduction.qmd`: To be written by all. 
- `sec1.qmd`: As an example Section of your website. It can be a page you all write together or an analysis for a subteam or an individual group memmber. 
- `sec2.ipynb`: As an example that you can also have pages built from `.ipynb` notebooks with quarto without the need of `R`.
- `conclusion.qmd`: To be written by all. 
- `dashboard.qmd`: Is a file for you to test quarto dashboards. It is optional. You can also delete it. 

To do for your team to structure your website further: 

- Read the file `_quarto.yml`. This file has all information that specifies this project as a Website. Read the comments in the file. Read the Quarto Documentation: [Quarto Websites](https://quarto.org/docs/websites/)
- You need to think about the number of Sections you have. This depends on your project. You need to create more files like `sec1.qmd` and `sec2.ipynb` and you need to add them to the contents in the file `_quarto.yml`. 

## Workflow for working on your website content

(For positron)

- Clone the repository.
- Open the file `index.qmd` (or any other `.qmd` file)
- Click preview. This should render all files, not just the file you are using!
- Work on your files and write your data science analysis as usual. 

## Workflow for documenting

According to the Syllabus, you have to provide a Workplan and **Team Member Contribution** reports three times. These reports have to be in the file `WORKPLAN.md`. 

- You can edit `WORKPLAN.md` in the browser on `GitHub.com`. (RECOMMENDED)
- You can also edit the files on you local machines but then you have to push them to GitHub. *Important:* The *.md*-files need not be rendered! These files are in the format *GitHub-flavored markdown* (`gfm`). This file format is for direct view on `GitHub.com` not for the creation of HTML files. 

## Workflow for collaborative git

- Remember besides `push`ing your work, you also need to `pull` the work of others. `git` will alert you probably. 
- When git problems appear, try to solve them, and use this occasion, to learn more about the concepts of git. If you cannot solve them, reach out to your colleague or the instructor, and find another way to continue your data science work!
- Remember: Sometimes the [Burn it all down](https://happygitwithr.com/burn) approach is an effective way to get into work mode again with git.
- Communicate with your team! Do some pushing an pulling while in a joint work session to understand how it works. Make a plan who works on what files or in what sections when to avoid too many git conflicts!  

## Publication

- This repository is private in its initial state. So, when you render the website it can only be viewed locally. 
- Your project can stay private forever and be deleted after submission and grading, but it can also be published as a website and be part of your portfolios. The instructor supports and recommends it. 
- Both ways have some inplications.

Public Website (Recommended): 

- It is recommended to start immediately with a public repository and setup the public Website with [GitHub Pages](https://docs.github.com/en/pages). That way you have a direct feedback how your work looks on the internet. (Just that the website is published does not mean that many people visit it, although they could.)
- This is how it looks on the Internet in its [initial state](https://janlorenz.github.io/DataScienceLab_Template/)
- Inform the instructor if you have group consensus that you work in public, then the repository is made public. 
- Follow the instructions to publish with GitHub Pages from the Quarto Documentation: [https://quarto.org/docs/publishing/github-pages.html]. 
    - The docs explain three ways. A good way is the second one [Publish Command](https://quarto.org/docs/publishing/github-pages.html#publish-command). Follow the instructions carefully, in particular the section [Source branch](https://quarto.org/docs/publishing/github-pages.html#source-branch). The basic idea is: You create a branch `gh-pages` which is solely used for the `_site` data of the website and then tell GitHub pages to take the website from there. Once the `gh-pages` branch is created you go back to the `main` branch and care anymore about the other. The command `quarto publish gh-pages` will update the website. 
- The instructor will assess your work mainly by looking at your website. 

Private Website for Submission: 

- When you want to stay private, the instructor cannot look at your website from the internet. A way to see your work is needed. 
- Contact the instructor during a regular meeting before submission to clarify the way of delivery! 


## Notes

- You can also work in RStudio or VSCode with quarto websites. Your core commands are `quarto preview` and `quarto render`
- Making the website work can be confusing when you do it the first time. This should not stop you from working out your data analysis. Composing the website can also be solved later! 
- You can design your site in your way, if you are interested to read a bit into it! 





