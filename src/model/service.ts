
interface IService{
    translate(text:string, from:string, to:string):Promise<string>;
}

export default IService;