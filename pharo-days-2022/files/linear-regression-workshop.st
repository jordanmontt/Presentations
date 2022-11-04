"Load the dataset"
lilleHousesDatasets := AIDatasets loadLilleHousing.
"Deal with the empty columns"
lilleHousesDatasets
    column: 'Rooms' 
    put: (lilleHousesDatasets column: 'Rooms') replaceNilsWithAverage.
lilleHousesDatasets
    toColumn: 'Rooms'
    applyElementwise: [ :each | each asInteger ].

"Normalizing the data frames"
normalizedDF := lilleHousesDatasets normalized.

"SEPARATING THE DATA"
"Dividing into test and training"
partitioner := AIRandomPartitioner new.
subsets := partitioner split: normalizedDF withProportions: #(0.75 0.25).
lilleHousesTrainDataset := subsets first.
lilleHousesTestDataset := subsets second.

"Separating between X and Y"
lilleHousesDatasets columnNames. "#('Area' 'Rooms' 'Price')"

xTrain := lilleHousesTrainDataset columns: #( 'Area' 'Rooms' ).
yTrain := lilleHousesTrainDataset column: 'Price'.

xTest := lilleHousesTestDataset columns: #( 'Area' 'Rooms' ).
yTest := lilleHousesTestDataset column:  'Price'.


"Converting the DataFrame into an array of arrays For using it in the linear model.
For now, the linear model does not work on a DataFrame."
xTrain := xTrain asArrayOfRows.
yTrain := yTrain asArray.

xTest := xTest asArrayOfRows.
yTest := yTest asArray.


"TRAINING THE MACHINE LEARNING MODEL"
"Training the linear regression model"
linearRegression := AILinearRegression
    learningRate: 0.5
    maxIterations: 5000.

linearRegression fitX: xTrain y: yTrain.

yPredicted := linearRegression predict: xTest.


"COMPUTING METRICS"
"Computing the accuracy of the logistic regression model"
metric := AIR2Score new.

r2Score := (metric computeForActual: yTest predicted: yPredicted) asFloat.