
interface IService{
    translate(text:string, language:string):Promise<string>;
}

export default IService;