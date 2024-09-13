import IService from "../model/service";
import ITransport from "../model/transport";
import transport from "../transport/Transport";


class TranslateService implements IService{
    
    private static instance:TranslateService;
 
    private transport:ITransport;
    
    private constructor(transport: ITransport){
        this.transport = transport;
    }

    public static getInstance(transport: ITransport): TranslateService {
        if (!TranslateService.instance) {
            TranslateService.instance = new TranslateService(transport);
        }
        return TranslateService.instance;
    }
    
    public async translate(text:string, from:string, to:string):Promise<string>{
        const response: string = await this.transport.post(text, from, to);
        return response;
    }
   
}

export default TranslateService.getInstance(transport);
