# Get CRC data sets from curatedMetagenomicData

# Install necessary pacakges
if(!require(curatedMetagenomicData, quietly = TRUE)){
  if(!require(BiocManager, quietly = TRUE)){
    install.packages("BiocManager")
  }
  library('BiocManager')
  BiocManager::install("curatedMetagenomicData")
}

if(!require(dplyr, quietly = TRUE)){
  install.packages('dplyr')
}

if(!require(feather, quietly = TRUE)){
  install.packages('feather')
}

if(!require(Matrix, quietly = TRUE)){
  install.packages('Matrix')
}

if(!require(jsonlite, quietly = TRUE)){
  install.packages('jsonlite')
}

# Load packages
library('curatedMetagenomicData')
library('dplyr')
library('feather') 
library('Matrix')
library('jsonlite')

# Create directory to save files into
save.dir <- 'CMD_files' # Can change path to save
if(!dir.exists(save.dir)){
  dir.create(save.dir)
}

# Get list of CRC studies, excluding HanniganGD_2017
study.list <- sampleMetadata %>% 
  filter(study_condition == 'CRC') %>%
  filter(study_name != 'HanniganGD_2017') %>% 
  pull(study_name) %>%
  unique()
print(sprintf('CRC studies: %s', paste(study.list, collapse=', ')))
  
# Download data from each study and save as feather file
for(study in study.list){
  print(paste('Getting data for ', study))
    
  # Get taxonomic abundances as counts
  taxa.list <- curatedMetagenomicData(paste(study, 'relative_abundance', sep='.'), 
                                            dryrun=FALSE, 
                                            counts=TRUE)
  TSE <- taxa.list[[1]]
  
  # Get metadata and keep only control and CRC samples
  meta.data <- as.data.frame(colData(TSE))
  allowable.condition <- c('control', 'CRC')
  meta.data <- meta.data %>% filter(study_condition %in% allowable.condition)
  
  # Correct mislabeled sample in YuJ_2015
  if('SZAXPI003427-1' %in% rownames(meta.data)){
    meta.data['SZAXPI003427-1','study_condition'] <- 'control'
    meta.data['SZAXPI003427-1','disease'] <- 'healthy'
  }
  
  # Get count table for control and CRC samples
  taxa.abun <- as.data.frame(t(assay(TSE))) 
  taxa.abun <- taxa.abun %>% subset(rownames(taxa.abun) %in% rownames(meta.data))
  print(sprintf('Dimensions of taxonomic abundance matrix: %d x %d',
                nrow(taxa.abun),
                ncol(taxa.abun)))
  
  # Get pathway abundances
  pathway.list <- curatedMetagenomicData(paste(study, 'pathway_abundance', sep='.'), 
                                         dryrun=FALSE, 
                                         counts=FALSE)
  
  # Keep only control/CRC samples, remove pathways attributed to a species (with '|')
  SE <- pathway.list[[1]]
  pathway.abun <- as.data.frame(t(assay(SE))) 
  pathway.abun <- pathway.abun %>% subset(rownames(pathway.abun) %in% rownames(meta.data))
  pathway.abun <- pathway.abun %>% select(!contains('|'))
  print(sprintf('Dimensions of pathway abundance matrix: %d x %d',
                nrow(pathway.abun),
                ncol(pathway.abun)))
  
  # Get gene-family abundances (feature table is a sparse matrix)
  gene.list <- curatedMetagenomicData(paste(study, 'gene_families', sep='.'), 
                                      dryrun=FALSE, 
                                      counts=FALSE)
  
  # Keep only control/CRC samples, remove pathways attributed to a species (with '|')
  SE <- gene.list[[1]]
  gf.sparse <- assay(SE)
  gf.sparse <- gf.sparse[(!grepl('|', rownames(gf.sparse), fixed=TRUE)),]
  gf.sparse <- gf.sparse[,colnames(gf.sparse) %in% rownames(meta.data)]
  print(sprintf('Dimensions of gene-family abundance matrix: %d x %d',
                nrow(gf.sparse),
                ncol(gf.sparse)))
  
  # Save files
  meta.file <- paste0(study, "-meta.feather")
  print(sprintf('Saving metadata to %s', paste(save.dir, meta.file, sep="/")))
  write_feather(meta.data, paste(save.dir, meta.file, sep="/"))
  
  taxa.file <- paste0(study, "-taxa-abun.feather")
  print(sprintf('Saving taxnomic abundances to %s', paste(save.dir, taxa.file, sep="/")))
  write_feather(taxa.abun, paste(save.dir, taxa.file, sep="/"))
  
  pathway.file <- paste0(study, "-pathway-abun.feather")
  print(sprintf('Saving pathway abundances to %s', paste(save.dir, pathway.file, sep="/")))
  write_feather(pathway.abun, paste(save.dir, pathway.file, sep="/"))
  
  gene.file <- paste0(study, "-gene-families.mtx")
  print(sprintf('Saving gene-family abundances to %s', paste(save.dir, gene.file, sep="/")))
  writeMM(gf.sparse, paste(save.dir, gene.file, sep="/"))
  
  gene.names <- paste0(study, "-gene-family-names.json")
  sprintf('Saving gene-family names to %s', paste(save.dir, gene.file, sep="/"))
  write_json(rownames(gf.sparse), paste(save.dir, gene.names, sep="/"))
}













    
    
    