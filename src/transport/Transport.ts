import axios from "axios";
import ITransport from "../model/transport";

class Transport implements ITransport{
    private static instance:Transport;
    private apiUrl:string;
   
    private constructor(url:string){
        this.apiUrl = url;
    }

    public static getInstance(url:string): Transport {
        if (!Transport.instance) {
            Transport.instance = new Transport(url);
        }
        return Transport.instance;
    }

    public async post(text:string, language:string):Promise<string> {
        try {
            
            // const response = await axios.post(this.apiUrl, {
            //     sign: sign,
            //     language: language,
            //     period: 'today'
            //   });
            // return response.data.horoscope;
            let newText = text;
            if(language === "Мансийский"){
                newText = text.toUpperCase();
            }
            return newText + "\n";
        
        } catch (error) {
            if (axios.isAxiosError(error)) {
                throw new Error(`transport.post: Error fetching data from API: ${error.response?.status} ${error.response?.statusText}`);
            } else {
                const err = error as Error;
                throw new Error(`transport.post: An unexpected error occurred: ${err.message}`);
            }
        }
    }
}

export default Transport.getInstance('https://poker247tech.ru/get_horoscope/')
