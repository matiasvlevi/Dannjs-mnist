const mnist = require('mnist');
const set = mnist.set(60000, 10000);
const trainData = set.training;
const testData = set.test;

const dn = require('dannjs');
const Dann = dn.dann;

//Creating the neural network:
const nn = new Dann(784,10);
nn.addHiddenLayer(128,'leakyReLU');
nn.addHiddenLayer(32,'leakyReLU');
nn.makeWeights();
nn.setLossFunction('bce');
nn.lr = 0.001;
nn.log();

totalEpoch = 20; // how many epochs to train the network.

//to define the model's guess into a numeric label:
function findLargest(arr) {
    let record = 0;
    let bestIndex = 0;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] > record) {
            record = arr[i];
            bestIndex = i;
        }
    }
    return bestIndex;
}

//train & test :
function train(batchPerEpoch,epoch) {
    for(let e = 0; e < epoch; e++) {
        console.log("Start Epoch: "+nn.epoch);
        let sum = 0;
        let batchlength = trainData.length/batchPerEpoch;
        for (let j=0; j < batchPerEpoch; j++) {
            let bsum = 0;
            for (let k = 0; k < batchlength; k++) {
                let data = trainData[(j*batchlength)+k];

                nn.backpropagate(data.input,data.output);

                bsum += nn.loss;
            }
            sum += bsum;
            let bavgLoss = bsum/batchlength;
            console.log("Batch: "+j+" AvgLoss: "+bavgLoss);
        }

        let avgloss = sum/trainData.length;
        let result = test();
        console.log("  ");
        console.log("Completed Epoch: "+nn.epoch+ "  Accuracy: "+result+" AvgLoss: "+avgloss);
        console.log("  ");
        nn.epoch++;
    }

}
function test() {
    let points = 0;
    for (data of testData) {
        let out = nn.feedForward(data.input,data.output);
        let predictedLabel = findLargest(out);
        let realLabel = findLargest(data.output);
        if (predictedLabel == realLabel) {
            points++;
        }
    }
    return points/testData.length;
}

//Training:
console.log('Training for '+totalEpoch+' epochs');
for (let i = 0; i < totalEpoch; i++) {
    train(100,1);
}
//Saving the model to ./savedDanns/mnist_model/
nn.save('mnist_model');
