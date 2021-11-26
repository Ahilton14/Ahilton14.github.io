# Prediciendo si los clientes abandonan las tarjetas de crédito 💳 

## 1 Descripción del Proyecto

### 1.1 Contexto del problema

Un gerente de banco se siente incómodo con que cada vez más clientes abandonen sus servicios de tarjeta de crédito. Realmente agradecerían que alguien pudiera predecir quién se verá afectado para poder acudir de manera proactiva al cliente para brindarle mejores servicios y cambiar las decisiones del cliente en la dirección opuesta.

### 1.2 El objetivo

Este proyecto se lleva a cabo en una secuencia de pasos, el primero de los cuales consiste en un análisis exploratorio, donde el objetivo es conocer el comportamiento de las variables y analizar atributos que indican una fuerte relación con la cancelación de los clientes del servicio de tarjetas de crédito. Tras la segunda parte, que consiste en aplicar técnicas de ingeniería de recursos, el tercer acto consiste en aplicar un algoritmo de aprendizaje automático para encontrar los mejores recursos para construir el modelo. Al final del proyecto, una vez finalizados todos los pasos, se desarrollará un modelo de aprendizaje automático, capaz de predecir, en base a los datos de un sistema, si un cliente dejará el servicio de tarjeta de crédito o no.

### 1.3 Conjunto de datos

Este conjunto de datos consta de 10,127 clientes que mencionan su edad, salario, estado civil, límite de tarjeta de crédito, categoría de tarjeta de crédito, etc.
Solo tenemos un 16,07% de clientes que han cancelado. Por lo tanto, es un poco difícil entrenar nuestro modelo para predecir la rotación de clientes.

#### variables más representativas:
+ **Attrition Flag:** Esta es nuestra variable objetivo, significa si nuestro cliente decidió dejar la organización o si existe una alta probabilidad de que el cliente se vaya.
+ **Gender:** Masculino o femenino.
+ **Customer age:** Edad del cliente
+ **Income category:** A qué categoría de ingresos pertenece el cliente.
+ **Card category:** Categoría de tarjeta que tiene el cliente.
+ **Months Inactive:** Meses de inactividad de usar la tarjeta de crédito.
+ **Credit Limit:** Límite de crédito que el cliente tiene actualmente.
+ **Total Revolving Balance:** La parte no pagada que se transfiere al mes siguiente cuando un cliente no paga.
+ **Average Utilization Ratio:** Mide la cantidad de crédito que está usando en comparación con la cantidad que tiene disponible.
+ **Open to buy:** La cantidad de crédito disponible en un momento dado en la cuenta del titular de una tarjeta de crédito. Por lo tanto, el promedio de apertura para comprar es el crédito promedio disponible asignado a un cliente específico.



```R
#Cargamos las librerias a usar en este proyecto
library(randomForest)
library(highcharter)
library(billboarder)
library(stringr)
library(ggplot2)
library(xgboost)
library(viridis)
library(timetk)
library(tidyr)
library(MASS)
library(dplyr)
library(caret)
library(DMwR)
library(ROSE)
library(glue)
library(pROC)
library(DT)
```


```R
#Cargando el conjunto de datos
BankChurners <- read.csv("./BankChurners.csv")
```


```R
# Eliminar columnas que no se utilizarán durante el análisis
BankChurners <- select(BankChurners, -c("CLIENTNUM", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"))

#Observamos las primeras 6 filas de la variable
head(BankChurners)
```


<table class="dataframe">
<caption>A data.frame: 6 × 20</caption>
<thead>
	<tr><th></th><th scope=col>Attrition_Flag</th><th scope=col>Customer_Age</th><th scope=col>Gender</th><th scope=col>Dependent_count</th><th scope=col>Education_Level</th><th scope=col>Marital_Status</th><th scope=col>Income_Category</th><th scope=col>Card_Category</th><th scope=col>Months_on_book</th><th scope=col>Total_Relationship_Count</th><th scope=col>Months_Inactive_12_mon</th><th scope=col>Contacts_Count_12_mon</th><th scope=col>Credit_Limit</th><th scope=col>Total_Revolving_Bal</th><th scope=col>Avg_Open_To_Buy</th><th scope=col>Total_Amt_Chng_Q4_Q1</th><th scope=col>Total_Trans_Amt</th><th scope=col>Total_Trans_Ct</th><th scope=col>Total_Ct_Chng_Q4_Q1</th><th scope=col>Avg_Utilization_Ratio</th></tr>
	<tr><th></th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>Existing Customer</td><td>45</td><td>M</td><td>3</td><td>High School</td><td>Married</td><td>$60K - $80K   </td><td>Blue</td><td>39</td><td>5</td><td>1</td><td>3</td><td>12691</td><td> 777</td><td>11914</td><td>1.335</td><td>1144</td><td>42</td><td>1.625</td><td>0.061</td></tr>
	<tr><th scope=row>2</th><td>Existing Customer</td><td>49</td><td>F</td><td>5</td><td>Graduate   </td><td>Single </td><td>Less than $40K</td><td>Blue</td><td>44</td><td>6</td><td>1</td><td>2</td><td> 8256</td><td> 864</td><td> 7392</td><td>1.541</td><td>1291</td><td>33</td><td>3.714</td><td>0.105</td></tr>
	<tr><th scope=row>3</th><td>Existing Customer</td><td>51</td><td>M</td><td>3</td><td>Graduate   </td><td>Married</td><td>$80K - $120K  </td><td>Blue</td><td>36</td><td>4</td><td>1</td><td>0</td><td> 3418</td><td>   0</td><td> 3418</td><td>2.594</td><td>1887</td><td>20</td><td>2.333</td><td>0.000</td></tr>
	<tr><th scope=row>4</th><td>Existing Customer</td><td>40</td><td>F</td><td>4</td><td>High School</td><td>Unknown</td><td>Less than $40K</td><td>Blue</td><td>34</td><td>3</td><td>4</td><td>1</td><td> 3313</td><td>2517</td><td>  796</td><td>1.405</td><td>1171</td><td>20</td><td>2.333</td><td>0.760</td></tr>
	<tr><th scope=row>5</th><td>Existing Customer</td><td>40</td><td>M</td><td>3</td><td>Uneducated </td><td>Married</td><td>$60K - $80K   </td><td>Blue</td><td>21</td><td>5</td><td>1</td><td>0</td><td> 4716</td><td>   0</td><td> 4716</td><td>2.175</td><td> 816</td><td>28</td><td>2.500</td><td>0.000</td></tr>
	<tr><th scope=row>6</th><td>Existing Customer</td><td>44</td><td>M</td><td>2</td><td>Graduate   </td><td>Married</td><td>$40K - $60K   </td><td>Blue</td><td>36</td><td>3</td><td>1</td><td>2</td><td> 4010</td><td>1247</td><td> 2763</td><td>1.376</td><td>1088</td><td>24</td><td>0.846</td><td>0.311</td></tr>
</tbody>
</table>




```R
#Asignamos a otra variable para ver la correlación de todas las variables
BankChurners_cor <- BankChurners
# Reemplazo de los elementos de la variable Attrition_Flag
sub_target <- function(x){
        if(x == "Existing Customer"){
                return(0)
        } else {
                return(1)
        }
}

BankChurners_cor$Attrition_Flag <- sapply(BankChurners_cor$Attrition_Flag, sub_target)

# Reemplazo de los elementos de la variable Attrition_Flag
sub_target1 <- function(x){
  if(x == "Existing.Customer"){
    return(0)
  } else {
    return(1)
  }
}
```

## 2 Análisis exploratorio de datos (EDA)
Como se indica en la "Descripción del proyecto", este paso tiene como objetivo descubrir los principales elementos responsables de la cancelación o no cancelación de los clientes del servicio de tarjeta de crédito. Para que esta primera sesión se lleve a cabo con éxito, aplicaré algunas técnicas estadísticas (Análisis Descriptivo) y visualización, que proporcionarán insights importantes y satisfactorios para continuar con el resto del análisis.

### 2.1 Análisis de correlación - Spearman
Para que este análisis no sea demasiado extenso, aplicaré la prueba estadística no paramétrica de Spearman, obteniendo así el coeficiente de correlación, que mide la dependencia estadística entre dos variables. De esta forma, podemos verificar desde el principio qué variables deben recibir más atención, ahorrando tiempo en el análisis de variables que no tienen una fuerte influencia en la tasa de clientes que abandonan el servicio de tarjetas de crédito.

#### Cómo calcular el coeficiente de correlación de Spearman
<img src="images/Spearman_correlation_coefficient.png" alt="New app" title="New app" width=300px height=100px/>
n = Número de puntos de datos para las dos variables

di = Diferencia en el alcance del elemento "n"

#### Interpretación del coeficiente de correlación de Spearman

El coeficiente de Spearman, ⍴, puede tener un valor entre +1 y -1 donde:

+ $\rho = +1$ → Significa una asociación de clasificación perfecta.

+ $\rho = 0$   →  Significa que no hay asociación de clasificación.

+ $\rho = -1$ → Significa una asociación negativa perfecta entre los intervalos.


```R
## Obtención de la matriz de correlaciones con el método de Spearman
cor_spearman <- cor(BankChurners_cor[, sapply(BankChurners_cor, is.numeric)], method = 'spearman')
```


```R
#Observamos la correkación de las variables
cor_spearman
```


<table class="dataframe">
<caption>A matrix: 15 × 15 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>Attrition_Flag</th><th scope=col>Customer_Age</th><th scope=col>Dependent_count</th><th scope=col>Months_on_book</th><th scope=col>Total_Relationship_Count</th><th scope=col>Months_Inactive_12_mon</th><th scope=col>Contacts_Count_12_mon</th><th scope=col>Credit_Limit</th><th scope=col>Total_Revolving_Bal</th><th scope=col>Avg_Open_To_Buy</th><th scope=col>Total_Amt_Chng_Q4_Q1</th><th scope=col>Total_Trans_Amt</th><th scope=col>Total_Trans_Ct</th><th scope=col>Total_Ct_Chng_Q4_Q1</th><th scope=col>Avg_Utilization_Ratio</th></tr>
</thead>
<tbody>
	<tr><th scope=row>Attrition_Flag</th><td> 1.00000000</td><td> 0.017508276</td><td> 0.020983325</td><td> 0.015299580</td><td>-0.149674044</td><td> 0.171838862</td><td> 0.18903770</td><td>-0.050909869</td><td>-0.240551008</td><td> 0.027500282</td><td>-0.101962046</td><td>-0.22378218</td><td>-0.37611517</td><td>-0.312058869</td><td>-0.240385406</td></tr>
	<tr><th scope=row>Customer_Age</th><td> 0.01750828</td><td> 1.000000000</td><td>-0.143583439</td><td> 0.768900951</td><td>-0.014495453</td><td> 0.044389284</td><td>-0.01439915</td><td> 0.002435331</td><td> 0.013550803</td><td>-0.002145866</td><td>-0.070537635</td><td>-0.03872595</td><td>-0.05385090</td><td>-0.040285029</td><td> 0.010562202</td></tr>
	<tr><th scope=row>Dependent_count</th><td> 0.02098332</td><td>-0.143583439</td><td> 1.000000000</td><td>-0.114844815</td><td>-0.035725507</td><td>-0.009174091</td><td>-0.04131038</td><td> 0.050695930</td><td>-0.003573976</td><td> 0.054436636</td><td>-0.026267383</td><td> 0.05784722</td><td> 0.05289694</td><td> 0.009413946</td><td>-0.034930147</td></tr>
	<tr><th scope=row>Months_on_book</th><td> 0.01529958</td><td> 0.768900951</td><td>-0.114844815</td><td> 1.000000000</td><td>-0.013973152</td><td> 0.057372365</td><td>-0.00828976</td><td> 0.006869865</td><td> 0.006289165</td><td> 0.007742943</td><td>-0.054364005</td><td>-0.02912749</td><td>-0.03877150</td><td>-0.033841635</td><td>-0.003643631</td></tr>
	<tr><th scope=row>Total_Relationship_Count</th><td>-0.14967404</td><td>-0.014495453</td><td>-0.035725507</td><td>-0.013973152</td><td> 1.000000000</td><td>-0.006644197</td><td> 0.06095406</td><td>-0.059278776</td><td> 0.011651022</td><td>-0.070821929</td><td> 0.025688554</td><td>-0.27911282</td><td>-0.22680752</td><td> 0.024238271</td><td> 0.065487167</td></tr>
	<tr><th scope=row>Months_Inactive_12_mon</th><td> 0.17183886</td><td> 0.044389284</td><td>-0.009174091</td><td> 0.057372365</td><td>-0.006644197</td><td> 1.000000000</td><td> 0.03033080</td><td>-0.027575460</td><td>-0.042543944</td><td>-0.015667979</td><td>-0.018773836</td><td>-0.03194521</td><td>-0.05085203</td><td>-0.046538490</td><td>-0.026558795</td></tr>
	<tr><th scope=row>Contacts_Count_12_mon</th><td> 0.18903770</td><td>-0.014399151</td><td>-0.041310376</td><td>-0.008289760</td><td> 0.060954058</td><td> 0.030330803</td><td> 1.00000000</td><td> 0.022717471</td><td>-0.044787770</td><td> 0.033264664</td><td>-0.020885933</td><td>-0.16737201</td><td>-0.16841280</td><td>-0.093310854</td><td>-0.058714607</td></tr>
	<tr><th scope=row>Credit_Limit</th><td>-0.05090987</td><td> 0.002435331</td><td> 0.050695930</td><td> 0.006869865</td><td>-0.059278776</td><td>-0.027575460</td><td> 0.02271747</td><td> 1.000000000</td><td> 0.131124749</td><td> 0.931430934</td><td> 0.021288663</td><td> 0.02840705</td><td> 0.03422192</td><td>-0.011408661</td><td>-0.416959382</td></tr>
	<tr><th scope=row>Total_Revolving_Bal</th><td>-0.24055101</td><td> 0.013550803</td><td>-0.003573976</td><td> 0.006289165</td><td> 0.011651022</td><td>-0.042543944</td><td>-0.04478777</td><td> 0.131124749</td><td> 1.000000000</td><td>-0.154164539</td><td> 0.036128948</td><td> 0.01766464</td><td> 0.04018485</td><td> 0.078223516</td><td> 0.708607200</td></tr>
	<tr><th scope=row>Avg_Open_To_Buy</th><td> 0.02750028</td><td>-0.002145866</td><td> 0.054436636</td><td> 0.007742943</td><td>-0.070821929</td><td>-0.015667979</td><td> 0.03326466</td><td> 0.931430934</td><td>-0.154164539</td><td> 1.000000000</td><td> 0.007040097</td><td> 0.02226666</td><td> 0.02157813</td><td>-0.040196481</td><td>-0.685716162</td></tr>
	<tr><th scope=row>Total_Amt_Chng_Q4_Q1</th><td>-0.10196205</td><td>-0.070537635</td><td>-0.026267383</td><td>-0.054364005</td><td> 0.025688554</td><td>-0.018773836</td><td>-0.02088593</td><td> 0.021288663</td><td> 0.036128948</td><td> 0.007040097</td><td> 1.000000000</td><td> 0.13458039</td><td> 0.08535425</td><td> 0.301981265</td><td> 0.032509208</td></tr>
	<tr><th scope=row>Total_Trans_Amt</th><td>-0.22378218</td><td>-0.038725952</td><td> 0.057847218</td><td>-0.029127490</td><td>-0.279112823</td><td>-0.031945208</td><td>-0.16737201</td><td> 0.028407047</td><td> 0.017664639</td><td> 0.022266656</td><td> 0.134580390</td><td> 1.00000000</td><td> 0.87972541</td><td> 0.222688253</td><td> 0.019351116</td></tr>
	<tr><th scope=row>Total_Trans_Ct</th><td>-0.37611517</td><td>-0.053850903</td><td> 0.052896944</td><td>-0.038771500</td><td>-0.226807516</td><td>-0.050852032</td><td>-0.16841280</td><td> 0.034221920</td><td> 0.040184852</td><td> 0.021578130</td><td> 0.085354253</td><td> 0.87972541</td><td> 1.00000000</td><td> 0.233448255</td><td> 0.040399142</td></tr>
	<tr><th scope=row>Total_Ct_Chng_Q4_Q1</th><td>-0.31205887</td><td>-0.040285029</td><td> 0.009413946</td><td>-0.033841635</td><td> 0.024238271</td><td>-0.046538490</td><td>-0.09331085</td><td>-0.011408661</td><td> 0.078223516</td><td>-0.040196481</td><td> 0.301981265</td><td> 0.22268825</td><td> 0.23344826</td><td> 1.000000000</td><td> 0.093993828</td></tr>
	<tr><th scope=row>Avg_Utilization_Ratio</th><td>-0.24038541</td><td> 0.010562202</td><td>-0.034930147</td><td>-0.003643631</td><td> 0.065487167</td><td>-0.026558795</td><td>-0.05871461</td><td>-0.416959382</td><td> 0.708607200</td><td>-0.685716162</td><td> 0.032509208</td><td> 0.01935112</td><td> 0.04039914</td><td> 0.093993828</td><td> 1.000000000</td></tr>
</tbody>
</table>




```R
#Visualizando con un mapa de calor la matriz de correlación con el método pearson
as.matrix(data.frame(cor_spearman)) %>% 
        round(3) %>% #round
        hchart() %>% 
        hc_add_theme(hc_theme_smpl()) %>%
        hc_title(text = "Coeficientes de correlación de Spearman", align = "center") %>% 
        hc_legend(align = "center") %>% 
        hc_colorAxis(stops = color_stops(colors = viridis::inferno(10))) %>%
        hc_plotOptions(
                series = list(
                        boderWidth = 0,
                        dataLabels = list(enabled = TRUE)))
```
