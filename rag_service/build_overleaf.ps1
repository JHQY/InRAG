# Create directories
New-Item -ItemType Directory -Force -Name "IRAG-report" | Out-Null
New-Item -ItemType Directory -Force -Path "IRAG-report/sections" | Out-Null
New-Item -ItemType Directory -Force -Path "IRAG-report/appendix" | Out-Null

# Write main.tex
@"
\documentclass[11pt]{article}
\input{preamble}
\addbibresource{refs.bib}
\title{IRAG: An Insurance Domain Multi-Modal Knowledge Base with Efficient RAG Interfaces}
\author{Your Name \and Collaborator Name}
\date{\today}

\newcommand{\mainbodyend}{%
  \clearpage
  \ifnum\value{page}>6
    \GenericWarning{}{[PAGE LIMIT] Main content exceeds 6 pages (\thepage).}
  \fi
}

\begin{document}
\maketitle

\begin{abstract}
\input{sections/00_abstract}
\end{abstract}

\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_system_architecture}
\input{sections/04_ingestion_indexing}
\input{sections/05_retrieval_api}
\input{sections/06_evaluation}
\input{sections/07_conclusion}

\mainbodyend

\printbibliography

\appendix
\clearpage
\section*{Appendix A: Datasets and Parsers}
\input{appendix/A_datasets_and_parsers}

\clearpage
\section*{Appendix B: Reproducibility Notes}
\input{appendix/B_reproducibility_notes}

\end{document}
"@ | Set-Content -Path "IRAG-report/main.tex"

# Write preamble.tex
@"
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{listings}
\lstset{basicstyle=\ttfamily\small,breaklines=true,frame=single,columns=fullflexible}
\usepackage[hidelinks]{hyperref}
\usepackage[backend=biber,style=ieee,sorting=nyt,maxbibnames=6]{biblatex}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\graphicspath{{figures/}}
"@ | Set-Content -Path "IRAG-report/preamble.tex"

# Write refs.bib
"% Add your references here" | Set-Content -Path "IRAG-report/refs.bib"

# Section placeholders
$sections = @(
    "00_abstract","01_introduction","02_related_work",
    "03_system_architecture","04_ingestion_indexing",
    "05_retrieval_api","06_evaluation","07_conclusion"
)

foreach ($s in $sections) {
    "% $s" | Set-Content -Path "IRAG-report/sections/$s.tex"
}

# Appendix placeholders
$appendix = @("A_datasets_and_parsers", "B_reproducibility_notes")

foreach ($a in $appendix) {
    "% $a" | Set-Content -Path "IRAG-report/appendix/$a.tex"
}

# Create ZIP
Compress-Archive -Path "IRAG-report" -DestinationPath "IRAG-report.zip" -Force

Write-Host "ZIP created: IRAG-report.zip"
