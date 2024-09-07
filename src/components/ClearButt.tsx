import clearImg from "./../assets/clear.svg";
import Button from "./Button";

interface ClearButtProps {
    tap: ()=>void;
}

const ClearButt = ({ tap }: ClearButtProps) => {

    return (
        <>
            <Button img={clearImg} tap={tap}  alt={"copy"}/>
        </>
        
    );
};

export default ClearButt;
