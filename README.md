# NMT4RDFS
Neural Machine Translation for RDFS reasoning: code and datasets for "Deep learning for noise-tolerant RDFS reasoning" http://www.semantic-web-journal.net/content/deep-learning-noise-tolerant-rdfs-reasoning-4

## Setting up environment
Create a virtual environment with python 3 

If using conda:
1. with GPU, run 

    ```conda create -n nmt4rdfs python=3.7 tensorflow-gpu"```
2. without GPU, run 
    
    ```conda create -n nmt4rdfs python=3.7 tensorflow"```

  Activate environment using: 
    
  ```conda activate nmt4rdfs```
    
    
Install requirements: 

   ```pip install -r requirements.txt```
    
## Creating graph words for LUBM1 dataset
 
 The first step to create the graph words is to collect the global resources i.e. classes, properties and properties groups as discussed in the paper. 
 You can either:
  1. Load the LUBM ontology from http://swat.cse.lehigh.edu/onto/univ-bench.owl into the repository specified in config.json. In this case the SPARQL queries defined in code/sparql are going to be used to collect the global resources.
  2. Use the pre-computed LUBM global resources from data/lubm1_intact/global_resources/: 
  
     ```
        cp data/lubm1_intact/global_resources/* data/lubm1_intact/encoding
        ```
  
 Generate the LUBM1 graph words using: 
  
  ```bash create_lubm_graph_words.sh ``` 
  
  
## Training phase: 
To train the graph words neural machine translator for LUBM1 using the configuration parameters at ``config.json`` run:

```cd code```

```python train_lubm.py``` 

## Inference phase:

To use the latest trained model for inference use:

```
python infer_lubm.py --input_graph=../data/lubm1_intact/graphs_with_descriptions/HTTP_www.Department0.University0.edu_AssistantProfessor0.nt
```

You can specify a different model using ```--model_path ```

## [Future updates]
- [ ] Instructions for generating graph words and training for the Scientists dataset
- [ ] Instructions to customize config.json for your dataset



