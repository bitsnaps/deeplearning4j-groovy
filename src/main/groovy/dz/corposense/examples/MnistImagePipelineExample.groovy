package dz.corposense.examples

import groovy.transform.CompileStatic
import groovy.util.logging.Slf4j
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.records.listener.impl.LogRecordListener
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.io.ClassPathResource
import org.nd4j.linalg.learning.config.Nesterovs


@Slf4j
@CompileStatic
class MnistImagePipelineExample {

    static void main(String[] args) {
        
        // image information
        // 28*28 grayscale
        // grayscale implies single channel
        int height = 28
        int width = 28
        int channels = 1
        int rngseed = 123
        Random randNumGen = new Random(rngseed)
        int batchSize = 128
        int outputNum = 10
        int numEpochs = 15
        double rate = 0.0015 // learning rate
        
        // Define the File Paths
        File trainData = new ClassPathResource('/mnist_png/training/').file
        File testData = new ClassPathResource('mnist_png/testing/').file

        // Define the FilePlist(PATH, ALLOWED FORMATS, random)
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)

        // Extract the parent path as the image label
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator()
        ImageRecordReader recordReader = new ImageRecordReader(height, width, labelMaker)

        // Initialize the record reader
        // add a listener, to extract the name
        recordReader.initialize(train)
        //recordReader.setListeners(new LogRecordListener())

        // DataSet Iterator
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum)

        // Scale pixel values to 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1)
        scaler.fit(dataIter)
        dataIter.setPreProcessor(scaler)

        // Show the formats of the data
/*        3.times {
            DataSet ds = dataIter.next()
            println( ds )
            println( dataIter.labels )
        }
*/

        // Build Neural Network Model
        log.info('*** Build Model ***')

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(rngseed)
            //.activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs(rate, 0.98d))
            .l2(1e-4d)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(height * width)
                .nOut(100)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .build())
            .layer(1, new OutputLayer.Builder()
                .nIn(100)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build())
            .pretrain(false)
            .backprop(true)
            .setInputType(InputType.convolutional(height, width, channels))
            .build()

        MultiLayerNetwork model = new MultiLayerNetwork(config)
        model.init()

        log.info("*** train model ****")

        numEpochs.times {
            log.info("Epoch " + it)
            model.fit(dataIter)
        }

        log.info("*** Evaluate model ****")

        recordReader.reset()
        recordReader.initialize(test)
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum)
        scaler.fit(testIter)
        testIter.setPreProcessor(scaler)

        // Create Eval object with 10 possible classes
        Evaluation eval = new Evaluation(outputNum)


        testIter.each {
            INDArray output = model.output(it.featureMatrix)
            eval.eval( it.labels, output)
        }

        log.info(eval.stats())

        log.info("*** Finish ***")


    }

}
