from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch
import numpy as np
import math

class SamplingDataset(object):
  def __init__(self,conf):
    self.num_classes = conf.num_classes
    self.dim = conf.dim
    self.num_labels = conf.num_labels
    self.mu, self.labels = self._get_data()

  def _get_data(self):
    mu = torch.normal(mean=0, std=math.sqrt(1/self.dim), size=(self.num_classes,self.dim))
    labels = torch.randint(self.num_labels, size=(self.num_classes,1))
    return mu, labels

class SamplingLoader(DataLoader):

  def __init__(self,conf, dataset):
    self.dataset = dataset
    self.mu, self.labels = self.dataset.mu, self.dataset.labels
    self.data_type = conf.data_type
    self.num_seq = conf.num_seq
    self.alpha = conf.alpha
    self.num_classes = conf.num_classes
    self.num_labels = conf.num_labels
    self.ways = conf.ways
    self.p_bursty = conf.p_bursty
    self.p_icl = conf.p_icl
    self.eps = conf.eps
    self.dim = conf.dim
    if self.ways != 0:
      assert self.num_seq % self.ways == 0
    if self.ways == 0:
      self.p_bursty = 0
    prob = np.array([1/((k+1)**self.alpha) for k in range(self.num_classes)])
    self.prob = prob/prob.sum()

  def get_seq(self):
    while True:
      if self.data_type=="bursty":
        if self.p_icl > np.random.rand():
            # choise few shot example
            num_few_shot_class = self.num_seq//self.ways
            mus, labels = self._get_novel_class_seq(num_few_shot_class)
            # mus = self.mu[few_shot_class]
            mus = np.repeat(mus, self.ways, axis=0) # expand ways
            # labels = self.labels[few_shot_class]
            labels = np.repeat(labels, self.ways, axis=0) # expand ways
            classes = np.arange(num_few_shot_class)
            classes = np.repeat(classes, self.ways)
            # add noise
            x = self.add_noise(mus)
            # permutation shuffle
            ordering = np.random.permutation(self.num_seq)
            mus = mus[ordering]
            x = x[ordering]
            labels = labels[ordering]
            classes = classes[ordering]
            # select query labels
            query_class_idx = np.random.choice(len(classes), 1)
            query_class = classes[query_class_idx]
            query_label = labels[query_class_idx]
            query_mu = mus[query_class_idx]
            query_x = self.add_noise(query_mu)
            # concat
            x = torch.cat([x, query_x])
            labels = torch.cat([labels.flatten(), query_label.flatten()])
            
            yield {
                "examples":x.to(torch.float32),
                "labels":labels,
                "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
            }
            
        else:
          if self.p_bursty > np.random.rand():
            # choise few shot example
            num_few_shot_class = self.num_seq//self.ways
            few_shot_class = np.random.choice(self.num_classes, num_few_shot_class, replace=False)
            mus = self.mu[few_shot_class]
            mus = np.repeat(mus, self.ways, axis=0) # expand ways
            labels = self.labels[few_shot_class]
            labels = np.repeat(labels, self.ways, axis=0) # expand ways
            classes = np.repeat(few_shot_class, self.ways)
            # add noise
            x = self.add_noise(mus)
            # permutation shuffle
            ordering = np.random.permutation(self.num_seq)
            x = x[ordering]
            labels = labels[ordering]
            classes = classes[ordering]
            # select query labels
            query_class = np.random.choice(few_shot_class, 1)
            query_label = self.labels[query_class]
            query_mu = self.mu[query_class]
            query_x = self.add_noise(query_mu)
            # concat
            x = torch.cat([x, query_x])
            labels = torch.cat([labels.flatten(), query_label.flatten()])
            yield {
                "examples":x.to(torch.float32),
                "labels":labels,
                "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
            }
            
          else:
            # rank frequency
            classes = np.random.choice(self.num_classes, self.num_seq+1, p=self.prob)
            mus = self.mu[classes]
            labels = self.labels[classes]
            x = self.add_noise(mus)
            # permutation shuffle
            ordering = np.random.permutation(self.num_seq+1)
            x = x[ordering]
            labels = labels[ordering]
            classes = classes[ordering]

            yield {
                "examples":x.to(torch.float32),
                "labels":labels.flatten(),
                "classes" : torch.from_numpy(classes)
            }

      elif self.data_type == "no_support":
          # rank frequency
          classes = np.random.choice(self.num_classes, self.num_seq+1, p=self.prob)
          mus = self.mu[classes]
          labels = self.labels[classes]
          x = self.add_noise(mus)
          # permutation shuffle
          ordering = np.random.permutation(self.num_seq+1)
          x = x[ordering]
          labels = labels[ordering]
          classes = classes[ordering]

          yield {
              "examples":x.to(torch.float32),
              "labels":labels.flatten(),
              "classes" : torch.from_numpy(classes)
          }
          
      elif self.data_type == "holdout":
        # choise few shot example
        num_few_shot_class = self.num_seq//self.ways
        mus, labels = self._get_novel_class_seq(num_few_shot_class)
        # mus = self.mu[few_shot_class]
        mus = np.repeat(mus, self.ways, axis=0) # expand ways
        # labels = self.labels[few_shot_class]
        labels = np.repeat(labels, self.ways, axis=0) # expand ways
        classes = np.arange(num_few_shot_class)
        classes = np.repeat(classes, self.ways)
        # add noise
        x = self.add_noise(mus)
        # permutation shuffle
        ordering = np.random.permutation(self.num_seq)
        mus = mus[ordering]
        x = x[ordering]
        labels = labels[ordering]
        classes = classes[ordering]
        # select query labels
        query_class_idx = np.random.choice(len(classes), 1)
        query_class = classes[query_class_idx]
        query_label = labels[query_class_idx]
        query_mu = mus[query_class_idx]
        query_x = self.add_noise(query_mu)
        # concat
        x = torch.cat([x, query_x])
        labels = torch.cat([labels.flatten(), query_label.flatten()])
        
        yield {
            "examples":x.to(torch.float32),
            "labels":labels,
            "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
        }

      elif self.data_type == "flip":
        # choise few shot example
        num_few_shot_class = self.num_seq//self.ways
        few_shot_class = np.random.choice(self.num_classes, num_few_shot_class, replace=False)
        mus = self.mu[few_shot_class]
        mus = np.repeat(mus, self.ways, axis=0) # expand ways
        classes = np.repeat(few_shot_class, self.ways)
        # label flip
        labels = (self.labels[classes] + 1) % self.num_labels
        # add noise
        x = self.add_noise(mus)
        # permutation shuffle
        ordering = np.random.permutation(self.num_seq)
        x = x[ordering]
        labels = labels[ordering]
        classes = classes[ordering]
        # select query labels
        query_class = np.random.choice(few_shot_class, 1)
        query_label = (self.labels[query_class] + 1) % self.num_labels
        query_mu = self.mu[query_class]
        query_x = self.add_noise(query_mu)
        # concat
        x = torch.cat([x, query_x])
        labels = torch.cat([labels.flatten(), query_label.flatten()])
        
        yield {
            "examples":x.to(torch.float32),
            "labels":labels,
            "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
        }
    
  

  def add_noise(self, x):
    x = (x+self.eps*torch.normal(mean=0, std=math.sqrt(1/self.dim), size=(x.shape)))/(np.sqrt(1+self.eps**2))
    # x = (x+self.eps*np.random.normal(mean=0, std=np.sqrt(1/self.dim), size=(x.shape[0],1)))/(np.sqrt(1+self.eps**2))
    return x
  
  def _get_novel_class_seq(self,num_class):
    mu = torch.normal(mean=0, std=math.sqrt(1/self.dim), size=(num_class,self.dim))
    labels = torch.randint(self.num_labels, size=(num_class,1))
    return mu, labels

class IterDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator


    def __iter__(self):
        return self.generator()

    
class MultiTaskSamplingLoader(DataLoader):

  def __init__(self,conf, dataset):
    self.dataset = dataset
    self.mu, self.labels = self.dataset.mu, self.dataset.labels
    self.data_type = conf.data_type
    self.num_seq = conf.num_seq
    self.alpha = conf.alpha
    self.num_classes = conf.num_classes
    self.num_task = conf.num_tasks
    self.num_labels = conf.num_labels
    self.task_ways = conf.task_ways
    self.item_ways = conf.item_ways
    self.p_bursty = conf.p_bursty
    self.p_icl = conf.p_icl
    self.eps = conf.eps
    self.dim = conf.dim
    if self.item_ways != 0 or self.task_ways != 0:
      assert self.num_seq % self.item_ways == 0 and self.num_seq % self.task_ways == 0
    if self.item_ways == 0 or self.task_ways == 0:
      self.p_bursty = 0
    prob = np.array([1/((k+1)**self.alpha) for k in range(self.num_classes)])
    self.prob = prob/prob.sum()

  def get_seq(self):
    while True:
      if self.data_type=="bursty":
        if self.p_bursty > np.random.rand():
          # choise few shot tasks
          num_few_shot_task = self.num_seq//self.task_ways
          few_shot_task = np.random.choice(self.num_task, num_few_shot_task, replace=False)
          tasks = np.repeat(few_shot_task, self.task_ways, axis=0).reshape(-1,1)
          
          # choise few shot items
          num_few_shot_class = self.num_seq//self.item_ways
          few_shot_class = np.random.choice(self.num_classes, num_few_shot_class, replace=False)
          mus = self.mu[few_shot_class]
          mus = np.repeat(mus, self.item_ways, axis=0) # expand ways
          
          # choice few shot labels
          labels = self.labels[few_shot_class]
          labels = np.repeat(labels, self.item_ways, axis=0) # expand ways
        
          
          # classes 
          classes = np.repeat(few_shot_class, self.item_ways)
          # add noise
          x = self.add_noise(mus)
          # permutation shuffle
          ordering = np.random.permutation(self.num_seq)
          x = x[ordering]
          labels = labels[ordering]
          classes = classes[ordering]
          task_ordering = np.random.permutation(self.num_seq)
          tasks = tasks[task_ordering]
          
          labels = (labels + tasks) % self.num_labels
          
          # select query labels
          # query_class = np.random.choice(few_shot_class, 1)
          query_class = np.random.choice(self.num_classes, 1)
          query_task = np.random.choice(few_shot_task, 1)
          query_label = (self.labels[query_class] + query_task) % self.num_labels
          query_mu = self.mu[query_class]
          query_x = self.add_noise(query_mu)
          # concat
          x = torch.cat([x, query_x])
          labels = torch.cat([labels.flatten(), query_label.flatten()])
          tasks = torch.cat([torch.tensor(tasks).flatten(), torch.tensor(query_task).flatten()])
          
          yield {
              "tasks":tasks,
              "examples":x.to(torch.float32),
              "labels":labels,
              "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
          }
          
        else:
          # rank frequency
          num_few_shot_task = self.num_seq//self.task_ways
          few_shot_task = np.random.choice(self.num_task, num_few_shot_task, replace=False)
          tasks = np.repeat(few_shot_task, self.task_ways, axis=0).reshape(-1,1)
          
          classes = np.random.choice(self.num_classes, self.num_seq+1, p=self.prob)
          mus = self.mu[classes]
          labels = self.labels[classes]
          x = self.add_noise(mus)
          # permutation shuffle
          ordering = np.random.permutation(self.num_seq+1)
          x = x[ordering]
          labels = labels[ordering]
          labels = (labels + tasks) % self.num_labels
          classes = classes[ordering]
          tasks = tasks[ordering]

          yield {
              "tasks":tasks,
              "examples":x.to(torch.float32),
              "labels":labels.flatten(),
              "classes" : torch.from_numpy(classes)
          }

      elif self.data_type == "no_support":
          num_few_shot_task = self.num_seq//self.task_ways
          few_shot_task = np.random.choice(self.num_task, num_few_shot_task, replace=False)
          tasks = np.repeat(few_shot_task, self.task_ways, axis=0).reshape(-1,1)
        
          # rank frequency
          classes = np.random.choice(self.num_classes, self.num_seq, p=self.prob)
          mus = self.mu[classes]
          # random label
          labels = np.random.randint(self.num_labels, size=(self.num_seq,1))
          x = self.add_noise(mus)
          # permutation shuffle
          ordering = np.random.permutation(self.num_seq)
          x = x[ordering]
          labels = labels[ordering]
          classes = classes[ordering]
          tasks = tasks[ordering]
          
          # select query labels
          query_class = np.random.choice(self.num_classes, 1)
          query_task = np.random.choice(few_shot_task, 1)
          query_label = self.labels[query_class]
          query_label = (query_label + query_task) % self.num_labels
          query_mu = self.mu[query_class]
          query_mu = self.add_noise(query_mu)
          
          # concat
          x = torch.cat([x, query_mu])
          labels = torch.cat([torch.from_numpy(labels).flatten(), query_label.flatten()])
          tasks = torch.cat([torch.tensor(tasks).flatten(), torch.tensor(query_task).flatten()])
          classes = np.concatenate([classes, query_class])

          yield {
              "tasks": tasks,
              "examples":x.to(torch.float32),
              "labels":labels.flatten(),
              "classes" : torch.from_numpy(classes)
          }
          
      elif self.data_type == "holdout":
        # choise few shot tasks
        num_few_shot_task = self.num_seq//self.task_ways
        few_shot_task = np.random.choice(self.num_task, num_few_shot_task, replace=False)
        tasks = np.repeat(few_shot_task, self.task_ways, axis=0).reshape(-1,1)
        false_tasks = np.random.choice(self.num_task, 1, replace=False)
        # print(tasks.shape)
        
        # choise few shot items
        num_few_shot_class = self.num_seq//self.item_ways
        few_shot_class = np.random.choice(self.num_classes, num_few_shot_class, replace=False)
        mus = self.mu[few_shot_class]
        mus = np.repeat(mus, self.item_ways, axis=0) # expand ways
        
        # choice few shot labels
        labels = self.labels[few_shot_class]
        labels = np.repeat(labels, self.item_ways, axis=0) # expand ways
        
        classes = np.repeat(few_shot_class, self.item_ways)
        
        # add noise
        x = self.add_noise(mus)
        # permutation shuffle
        ordering = np.random.permutation(self.num_seq)
        mus = mus[ordering]
        x = x[ordering]
        labels = labels[ordering]
        classes = classes[ordering]
        tasks = tasks[ordering]
        
        labels = (labels + tasks) % self.num_labels
        
        # select query labels
        # query_class = np.random.choice(few_shot_class, 1)
        query_class = np.random.choice(self.num_classes, 1)
        query_task = np.random.choice(few_shot_task, 1)
        query_label = (self.labels[query_class] + query_task) % self.num_labels
        query_mu = self.mu[query_class]
        query_x = self.add_noise(query_mu)
        # concat
        x = torch.cat([x, query_x])
        labels = torch.cat([labels.flatten(), query_label.flatten()])
        tasks = torch.cat([torch.tensor(tasks).flatten(), torch.tensor(false_tasks).flatten()])
          
        yield {
            "tasks":tasks,
            "examples":x.to(torch.float32),
            "labels":labels,
            "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
        }

      elif self.data_type == "flip":
        # choise few shot example
        num_few_shot_class = self.num_seq//self.item_ways
        few_shot_class = np.random.choice(self.num_classes, num_few_shot_class, replace=False)
        mus = self.mu[few_shot_class]
        mus = np.repeat(mus, self.item_ways, axis=0) # expand ways
        classes = np.repeat(few_shot_class, self.item_ways)
        # label flip
        labels = (self.labels[classes] + 1) % self.num_labels
        # add noise
        x = self.add_noise(mus)
        # permutation shuffle
        ordering = np.random.permutation(self.num_seq)
        x = x[ordering]
        labels = labels[ordering]
        classes = classes[ordering]
        # select query labels
        query_class = np.random.choice(few_shot_class, 1)
        query_label = (self.labels[query_class] + 1) % self.num_labels
        query_mu = self.mu[query_class]
        query_x = self.add_noise(query_mu)
        # concat
        x = torch.cat([x, query_x])
        labels = torch.cat([labels.flatten(), query_label.flatten()])
        
        yield {
            "examples":x.to(torch.float32),
            "labels":labels,
            "classes" : torch.cat([torch.from_numpy(classes).flatten(), torch.from_numpy(query_class).flatten()])
        }
    
  

  def add_noise(self, x):
    x = (x+self.eps*torch.normal(mean=0, std=math.sqrt(1/self.dim), size=(x.shape)))/(np.sqrt(1+self.eps**2))
    # x = (x+self.eps*np.random.normal(mean=0, std=np.sqrt(1/self.dim), size=(x.shape[0],1)))/(np.sqrt(1+self.eps**2))
    return x
  
  def _get_novel_class_seq(self,num_class):
    mu = torch.normal(mean=0, std=math.sqrt(1/self.dim), size=(num_class,self.dim))
    labels = torch.randint(self.num_labels, size=(num_class,1))
    return mu, labels