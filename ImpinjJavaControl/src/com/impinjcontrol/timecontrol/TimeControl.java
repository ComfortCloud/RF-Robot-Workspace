package com.impinjcontrol.timecontrol;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;


public class TimeControl {
    /**
     * @param args
     */
    public static void main(String[] args) {
        BasicConfigurator.configure();

        // Only show root events from the base logger
        Logger.getRootLogger().setLevel(Level.ERROR);
        ReaderHandler example = new ReaderHandler();
        ReaderHandler.logger.setLevel(Level.INFO);

//    if  (args.length < 1) {
//        System.out.print("Must pass reader hostname or IP as agument 1");
//        System.exit(-1);
//    }

        example.connect("115.156.142.223");
        example.enableImpinjExtensions();
        example.factoryDefault();
        example.getReaderCapabilities();
        example.getReaderConfiguration();
        example.setReaderConfiguration();
        example.addRoSpec(true);
        example.enable();
        example.start();

        try {
            Thread.sleep(2000);
        } catch (InterruptedException ex) {
            ReaderHandler.logger.error("Sleep Interrupted");
        }

        example.stop();
        example.disconnect();
        System.exit(0);
    }
}