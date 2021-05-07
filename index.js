const func = require('./func.js');
const dataset = require('easy-mnist').makeData(60000,10000);

const dn = require('dannjs');
const Dann = dn.dann;
const Layer = dn.layer;

let epoch = JSON.parse(process.argv[2]);

//Creating a pooling layer to downsample (784 to 196):
let dsl = new Layer('avgpool',784,2,2);

//Creating the Deep Neural Network:
let nn = new Dann(196,11);
nn.addHiddenLayer(169,'leakyReLU');
nn.addHiddenLayer(81,'leakyReLU');
nn.addHiddenLayer(36,'leakyReLU');
nn.setLossFunction('bce');
nn.makeWeights(-0.5,0.5);
nn.lr = 0.000005;
nn.log();

//train & test :
function train_(batchPerEpoch,epoch) {
    let len = dataset.traindata.length;
    for(let e = 0; e < epoch; e++) {
        console.log("Start Epoch: "+nn.epoch);
        let sum = 0;
        let batchlength = len/batchPerEpoch;
        for (let j=0; j < batchPerEpoch; j++) {
            let bsum = 0;
            for (let k = 0; k < batchlength; k++) {
                let index = (j*batchlength)+k;
                //Downsampling 28x28 image to 14x14
                let downsampled = dsl.feed(dataset.traindata[index].image);
                //Adding one more value to the target array to allow for the model to recognise 'blank' images.
                let target = dataset.traindata[index].label;
                target[10] = 0;
                nn.backpropagate(downsampled,target);
                bsum += nn.loss;
            }
            //Training to allow for the model to recognise 'blank' images.
            for (let i = 0; i < batchlength/20; i++) {
                let downsampled = func.makeEmpty(196);
                let target = [0,0,0,0,0,0,0,0,0,0,1];
                nn.backpropagate(downsampled,target);
            }
            sum += bsum;
            let bavgLoss = bsum/batchlength;
            console.log("Batch: "+j+" AvgLoss: "+bavgLoss);
        }
        let avgloss = sum/len;
        //Testing model's accuracy
        let result = test_();
        console.log("  ");
        console.log("Completed Epoch: "+nn.epoch+ "  Accuracy: "+result+" AvgLoss: "+avgloss);
        console.log("  ");
        nn.epoch++;
    }
}
function test_() {
    let points = 0;
    let len = dataset.testdata.length;
    for (let i = 0; i < len; i++) {
        //Downsampling 28x28 image to 14x14
        let downsampled = dsl.feed(dataset.testdata[i].image);
        //finding the largest values of the model's output predictions.
        let output = nn.feedForward(downsampled);
        if (func.findLargest(output) == dataset.testdata[i].label.indexOf(1)) {
            points++;
        }
    }
    return points/len;
}

for (let i = 0; i < epoch; i++) {
    train_(100,1);
    nn.save('trainedMnistModel');
}
