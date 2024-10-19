library(tidyverse)


data <- read.csv('fricrot_data.csv')

data %>% pull(Marca) %>% unique() %>% length()
data %>% pull(Modelo) %>% unique() %>% length()

data %>% mutate(
  m_m = paste(Marca, Modelo)
) %>% pull(m_m) %>% unique() %>% length()

tabla = table(data$Marca, data$Modelo)
tabla = tabla %>% as.data.frame() %>% filter(Freq > 0)
tabla2 = table(tabla$Var2) %>% as.data.frame()

data %>% filter(Modelo == "AMAROK" & Anio == 2024) %>% View()

data %>% 
  group_by(Modelo) %>% 
  summarise(anios = n_distinct(Anio)) %>% 
  ggplot(aes(anios)) + geom_histogram(stat = "count") +
  theme_bw()
