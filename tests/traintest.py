import unittest
import msdnet
import numpy as np

class TestTraining(unittest.TestCase):

    def test_regr_training(self):
        dil = msdnet.dilations.IncrementDilations(3)
        n = msdnet.network.MSDNet(10, dil, 1, 1, gpu=False)
        n.initialize()
        tgt_im = np.zeros((1,128,128), dtype=np.float32)
        tgt_im[:,32:96,32:96] = 1
        inp_im = (tgt_im + np.random.normal(size=tgt_im.shape)).astype(np.float32)
        d = msdnet.data.ArrayDataPoint(inp_im, tgt_im)
        bprov = msdnet.data.BatchProvider([d,],1)
        loss = msdnet.loss.L2Loss()
        val = msdnet.validate.LossValidation([d,], loss=loss)
        t = msdnet.train.AdamAlgorithm(n, loss=loss)
        n.normalizeinout([d,])
        log1 = msdnet.loggers.ConsoleLogger()
        log2 = msdnet.loggers.FileLogger('test_log_regr.txt')
        log3 = msdnet.loggers.ImageLogger('test_log_regr')
        for i in range(10):
            t.step(n, bprov.getbatch())
            t.to_file('test_regr_params.h5')
            val.to_file('test_regr_params.h5')
            n.to_file('test_regr_params.h5')
            val.validate(n)
            log1.log(val)
            log2.log(val)
            log3.log(val)
        n_load = msdnet.network.MSDNet.from_file('test_regr_params.h5', gpu=False)
        n_load.forward(inp_im)
    
    def test_segm_training(self):
        dil = msdnet.dilations.IncrementDilations(3)
        n = msdnet.network.SegmentationMSDNet(10, dil, 1, 2, gpu=False)
        n.initialize()
        tgt_im = np.zeros((1,128,128), dtype=np.uint8)
        tgt_im[:,32:96,32:96] = 1
        inp_im = (tgt_im + np.random.normal(size=tgt_im.shape)).astype(np.float32)
        d = msdnet.data.ArrayDataPoint(inp_im, tgt_im)
        d = msdnet.data.OneHotDataPoint(d,[0,1])
        bprov = msdnet.data.BatchProvider([d,],1)
        loss = msdnet.loss.CrossEntropyLoss()
        val = msdnet.validate.LossValidation([d,], loss=loss)
        t = msdnet.train.AdamAlgorithm(n, loss=loss)
        n.normalizeinout([d,])
        log1 = msdnet.loggers.ConsoleLogger()
        log2 = msdnet.loggers.FileLogger('test_log_segm.txt')
        log3 = msdnet.loggers.ImageLabelLogger('test_log_segm')
        for i in range(10):
            t.step(n, bprov.getbatch())
            t.to_file('test_segm_params.h5')
            val.to_file('test_segm_params.h5')
            n.to_file('test_segm_params.h5')
            val.validate(n)
            log1.log(val)
            log2.log(val)
            log3.log(val)
        n_load = msdnet.network.SegmentationMSDNet.from_file('test_segm_params.h5', gpu=False)
        n_load.forward(inp_im)

if __name__ == '__main__':
    unittest.main()